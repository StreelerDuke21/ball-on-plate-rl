# Ball-on-Plate Reinforcement Learning (SAC)

Proyecto completo y portable para entrenar un agente **Soft Actor-Critic (SAC)** que equilibra una bolita en el centro de una placa controlada por dos servomotores.

Basado en el paper:

> Kóczi, D.; Németh, J.; Sárosi, J.  
> *Two-Dimensional Positioning with Machine Learning in Virtual and Real Environments*.  
> Electronics 2023, 12, 671.  
> https://www.mdpi.com/2079-9292/12/3/671

---

## Estructura del proyecto

```
ball_on_plate_rl/
├── src/
│   ├── ball_on_plate_env.py      # Entorno Gymnasium
│   ├── ball_tracker.py           # Visión por computador (OpenCV)
│   └── servo_controller.py       # Control de servomotores
├── scripts/
│   ├── train_sac.py              # Entrenamiento
│   ├── test_agent.py             # Evaluación de modelos
│   ├── test_camera.py            # Calibración de cámara
│   └── run_real.py               # Control real (modelo + cámara + servos)
├── arduino/
│   └── ball_on_plate_servos.ino  # Firmware Arduino para 2 servos
├── models/                       # Modelos guardados (local)
├── logs/                         # TensorBoard + evaluaciones (local)
├── requirements.txt
├── setup_venv.sh
└── README.md
```

---

## Instalación rápida

### Opción A – Script automático (Linux/macOS)

```bash
cd ball_on_plate_rl
chmod +x setup_venv.sh
./setup_venv.sh
```

### Opción B – Manual (Windows / Linux / macOS)

```bash
cd ball_on_plate_rl
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Entrenamiento (simulación)

```bash
python scripts/train_sac.py
python scripts/train_sac.py --timesteps 500000
```

Evaluar:

```bash
python scripts/test_agent.py
python scripts/test_agent.py --model models/best/best_model --episodes 20
```

---

## Visión con cámara

```bash
python scripts/test_camera.py
python scripts/test_camera.py --calibrate
python scripts/test_camera.py --roi --calibrate
```

Coloca la cámara en **vista aérea** (perpendicular a la placa).

---

## Control real (hardware)

Modo prueba (sin servos):

```bash
python scripts/run_real.py --servo mock --calibrate
```

Con Arduino por USB:

```bash
pip install pyserial
python scripts/run_real.py --servo serial --port COM3
```

Sube `arduino/ball_on_plate_servos.ino` al microcontrolador (protocolo: `S,<alpha>,<beta>\n`).

---

## Dependencias principales

- Python 3.10 – 3.12
- gymnasium, stable-baselines3, torch
- opencv-python, pyserial

## Licencia

Código libre para uso educativo y de investigación.
