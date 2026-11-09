"""
Control de servomotores para la placa Ball-on-Plate.

Soporta varios backends:
  - "mock"     -> solo imprime los angulos (para probar sin hardware)
  - "serial"   -> envia comandos por puerto serie a un Arduino / microcontrolador
  - "pca9685"  -> (opcional) PCA9685 por I2C (mas tipico en Raspberry Pi)

Protocolo serie esperado (Arduino):
  "S,<alpha_deg>,<beta_deg>\n"
  Ejemplo: S,3.50,-1.20\n
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseServoController(ABC):
    """Interfaz comun para cualquier controlador de servos."""

    def __init__(self, max_angle_deg: float = 12.0):
        self.max_angle_deg = max_angle_deg
        self._alpha = 0.0
        self._beta = 0.0

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def _send(self, alpha_deg: float, beta_deg: float):
        ...

    def set_angles(self, alpha_deg: float, beta_deg: float):
        """Establece los angulos de la placa (en grados)."""
        alpha_deg = float(max(-self.max_angle_deg, min(self.max_angle_deg, alpha_deg)))
        beta_deg = float(max(-self.max_angle_deg, min(self.max_angle_deg, beta_deg)))
        self._alpha = alpha_deg
        self._beta = beta_deg
        self._send(alpha_deg, beta_deg)

    def center(self):
        """Pone la placa horizontal."""
        self.set_angles(0.0, 0.0)

    @property
    def angles(self) -> Tuple[float, float]:
        return self._alpha, self._beta


class MockServoController(BaseServoController):
    """No mueve nada: solo muestra los angulos en consola."""

    def connect(self) -> bool:
        print("[MockServo] Conectado (modo simulacion)")
        return True

    def disconnect(self):
        print("[MockServo] Desconectado")

    def _send(self, alpha_deg: float, beta_deg: float):
        print(f"[MockServo] a={alpha_deg:+6.2f} deg   b={beta_deg:+6.2f} deg")


class SerialServoController(BaseServoController):
    """Envia comandos por puerto serie a un Arduino / ESP32 / similar."""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        max_angle_deg: float = 12.0,
        timeout: float = 1.0,
    ):
        super().__init__(max_angle_deg=max_angle_deg)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser = None

    def connect(self) -> bool:
        try:
            import serial
        except ImportError:
            print("Error: instala pyserial ->  pip install pyserial")
            return False

        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)
            print(f"[SerialServo] Conectado en {self.port} @ {self.baudrate}")
            self.center()
            return True
        except Exception as e:
            print(f"[SerialServo] No se pudo abrir {self.port}: {e}")
            self._ser = None
            return False

    def disconnect(self):
        if self._ser is not None:
            try:
                self.center()
                time.sleep(0.1)
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            print("[SerialServo] Desconectado")

    def _send(self, alpha_deg: float, beta_deg: float):
        if self._ser is None or not self._ser.is_open:
            return
        cmd = f"S,{alpha_deg:.2f},{beta_deg:.2f}\n"
        try:
            self._ser.write(cmd.encode("ascii"))
        except Exception as e:
            print(f"[SerialServo] Error al enviar: {e}")


def make_servo_controller(
    backend: str = "mock",
    port: str = "COM3",
    max_angle_deg: float = 12.0,
) -> BaseServoController:
    """Factory sencilla."""
    backend = backend.lower().strip()
    if backend == "mock":
        return MockServoController(max_angle_deg=max_angle_deg)
    if backend == "serial":
        return SerialServoController(port=port, max_angle_deg=max_angle_deg)
    raise ValueError(f"Backend de servos desconocido: {backend}. Usa 'mock' o 'serial'.")
