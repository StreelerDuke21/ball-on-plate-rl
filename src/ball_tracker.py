"""
Módulo de visión para tracking de la bolita con webcam.
Usa detección por color en espacio HSV (robusto y rápido).

Uso típico:
    tracker = BallTracker(camera_id=0, plate_size_m=0.20)
    tracker.calibrate()          # opcional: ajustar color en vivo
    x, y, found = tracker.get_position()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class BallTrackerConfig:
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30

    # Tamaño real de la placa (metros). Se usa para normalizar la posición.
    plate_half_size_m: float = 0.10          # ±10 cm → placa de 20 cm

    # Rango HSV por defecto (naranja / rojo-naranja típico de bolita)
    # Se puede recalibrar en runtime con calibrate()
    hsv_lower: Tuple[int, int, int] = (5, 100, 100)
    hsv_upper: Tuple[int, int, int] = (25, 255, 255)

    # Filtros de blob
    min_radius_px: int = 8
    max_radius_px: int = 80
    blur_ksize: int = 5

    # Suavizado de posición (filtro exponencial)
    smooth_alpha: float = 0.45


class BallTracker:
    """
    Detecta la bolita en la imagen de la webcam y devuelve su posición
    normalizada respecto al centro de la placa.
    """

    def __init__(self, config: Optional[BallTrackerConfig] = None, **kwargs):
        if config is None:
            config = BallTrackerConfig(**kwargs)
        self.cfg = config

        self.cap: Optional[cv2.VideoCapture] = None
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._last_found: bool = False

        # Región de interés (se puede ajustar para recortar solo la placa)
        self.roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

    # ------------------------------------------------------------------
    # Cámara
    # ------------------------------------------------------------------
    def open(self) -> bool:
        """Abre la cámara. Devuelve True si tuvo éxito."""
        self.cap = cv2.VideoCapture(self.cfg.camera_id, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
        if not self.cap.isOpened():
            # Reintento sin backend específico
            self.cap = cv2.VideoCapture(self.cfg.camera_id)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        return True

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __enter__(self):
        if not self.open():
            raise RuntimeError(f"No se pudo abrir la cámara {self.cfg.camera_id}")
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Detección
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[y : y + h, x : x + w]

        blurred = cv2.GaussianBlur(frame, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def _find_ball(self, hsv: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Devuelve (cx, cy, radius) en píxeles del frame (o del ROI),
        o None si no se detecta nada válido.
        """
        lower = np.array(self.cfg.hsv_lower, dtype=np.uint8)
        upper = np.array(self.cfg.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Limpieza morfológica
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Elegir el contorno más grande que parezca una bola
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if self.cfg.min_radius_px <= radius <= self.cfg.max_radius_px:
                if area > best_area:
                    best_area = area
                    best = (float(x), float(y), float(radius))

        return best

    def _pixel_to_normalized(self, cx: float, cy: float, frame_shape: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Convierte coordenadas de píxel al sistema normalizado del entorno:
        centro = (0, 0), bordes de la placa ≈ ±1.
        """
        h, w = frame_shape[:2]
        if self.roi is not None:
            # Ya estamos dentro del ROI
            pass

        # Origen en el centro de la imagen / ROI
        x_norm = (cx - w / 2) / (w / 2)
        y_norm = (cy - h / 2) / (h / 2)

        # Invertir Y porque en imagen Y crece hacia abajo
        y_norm = -y_norm

        return float(x_norm), float(y_norm)

    def get_position(self) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Lee un frame y devuelve:
            x_norm, y_norm, found

        x_norm, y_norm ∈ [-1.5 .. 1.5] aproximadamente (0 = centro).
        found = True si se detectó la bolita.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Cámara no abierta. Llama a open() primero.")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, False

        hsv = self._preprocess(frame)
        result = self._find_ball(hsv)

        if result is None:
            self._last_found = False
            return self._smooth_x, self._smooth_y, False

        cx, cy, radius = result
        x_norm, y_norm = self._pixel_to_normalized(cx, cy, frame.shape)

        # Suavizado exponencial
        alpha = self.cfg.smooth_alpha
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = x_norm, y_norm
        else:
            self._smooth_x = alpha * x_norm + (1 - alpha) * self._smooth_x
            self._smooth_y = alpha * y_norm + (1 - alpha) * self._smooth_y

        self._last_found = True
        return self._smooth_x, self._smooth_y, True

    def get_observation_components(self) -> Tuple[float, float, float, float]:
        """
        Devuelve (x_norm, y_norm, vx_approx, vy_approx) listos para
        construir la observación del agente (los primeros 4 valores).
        Velocidades se estiman de forma simple entre frames.
        """
        x, y, found = self.get_position()
        if x is None or y is None:
            return 0.0, 0.0, 0.0, 0.0

        # Estimación muy simple de velocidad (se puede mejorar con dt real)
        if not hasattr(self, "_prev_x"):
            self._prev_x, self._prev_y = x, y
            self._prev_t = time.time()
            return x, y, 0.0, 0.0

        now = time.time()
        dt = max(now - self._prev_t, 1e-3)
        vx = (x - self._prev_x) / dt
        vy = (y - self._prev_y) / dt

        self._prev_x, self._prev_y = x, y
        self._prev_t = now

        # Normalización similar a la del entorno simulado
        vx_norm = np.clip(vx / 1.5, -3.0, 3.0)
        vy_norm = np.clip(vy / 1.5, -3.0, 3.0)

        return float(x), float(y), float(vx_norm), float(vy_norm)

    # ------------------------------------------------------------------
    # Utilidades de calibración y visualización
    # ------------------------------------------------------------------
    def set_roi_interactive(self):
        """Permite seleccionar la región de la placa con el ratón."""
        if self.cap is None:
            raise RuntimeError("Cámara no abierta")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer frame")

        print("Selecciona el área de la placa (arrastrar) y pulsa ENTER / ESPACIO")
        roi = cv2.selectROI("Selecciona la placa", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Selecciona la placa")

        if roi[2] > 0 and roi[3] > 0:
            self.roi = tuple(map(int, roi))
            print(f"ROI configurado: {self.roi}")
        else:
            print("ROI no cambiado")

    def calibrate_color(self, window_name: str = "Calibracion HSV"):
        """
        Abre una ventana interactiva para ajustar los umbrales HSV
        hasta que la bolita se vea bien en la máscara.
        Pulsa 'q' para guardar y salir.
        """
        if self.cap is None:
            raise RuntimeError("Cámara no abierta")

        def nothing(x):
            pass

        cv2.namedWindow(window_name)
        cv2.createTrackbar("H min", window_name, self.cfg.hsv_lower[0], 179, nothing)
        cv2.createTrackbar("S min", window_name, self.cfg.hsv_lower[1], 255, nothing)
        cv2.createTrackbar("V min", window_name, self.cfg.hsv_lower[2], 255, nothing)
        cv2.createTrackbar("H max", window_name, self.cfg.hsv_upper[0], 179, nothing)
        cv2.createTrackbar("S max", window_name, self.cfg.hsv_upper[1], 255, nothing)
        cv2.createTrackbar("V max", window_name, self.cfg.hsv_upper[2], 255, nothing)

        print("Ajusta los sliders hasta ver solo la bolita. Pulsa 'q' para terminar.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            display = frame.copy()
            if self.roi is not None:
                x, y, w, h = self.roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_frame = frame[y : y + h, x : x + w]
            else:
                roi_frame = frame

            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            h_min = cv2.getTrackbarPos("H min", window_name)
            s_min = cv2.getTrackbarPos("S min", window_name)
            v_min = cv2.getTrackbarPos("V min", window_name)
            h_max = cv2.getTrackbarPos("H max", window_name)
            s_max = cv2.getTrackbarPos("S max", window_name)
            v_max = cv2.getTrackbarPos("V max", window_name)

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)

            result = self._find_ball(hsv)
            if result is not None:
                cx, cy, r = result
                if self.roi is not None:
                    cx += self.roi[0]
                    cy += self.roi[1]
                cv2.circle(display, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.circle(display, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            # Mostrar máscara en la esquina
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mh, mw = mask_bgr.shape[:2]
            scale = 0.35
            small = cv2.resize(mask_bgr, (int(mw * scale), int(mh * scale)))
            display[10 : 10 + small.shape[0], 10 : 10 + small.shape[1]] = small

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.cfg.hsv_lower = (h_min, s_min, v_min)
                self.cfg.hsv_upper = (h_max, s_max, v_max)
                print(f"Nuevos umbrales HSV: lower={self.cfg.hsv_lower}  upper={self.cfg.hsv_upper}")
                break

        cv2.destroyWindow(window_name)

    def preview(self, duration_sec: float = 30.0):
        """Muestra la detección en vivo durante unos segundos (útil para pruebas)."""
        if self.cap is None:
            raise RuntimeError("Cámara no abierta")

        t0 = time.time()
        while time.time() - t0 < duration_sec:
            ret, frame = self.cap.read()
            if not ret:
                break

            x, y, found = self.get_position()
            display = frame.copy()

            if self.roi is not None:
                rx, ry, rw, rh = self.roi
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)

            # Cruz del centro
            h, w = display.shape[:2]
            cv2.drawMarker(display, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            status = f"Ball: {'OK' if found else '---'}  x={x:.3f}  y={y:.3f}" if x is not None else "Ball: ---"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 0, 255), 2)

            cv2.imshow("Ball Tracker Preview", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Ball Tracker Preview")
