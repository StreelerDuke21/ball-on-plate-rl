#!/bin/bash
# ============================================================
#  Setup del entorno virtual – Ball-on-Plate RL
#  Ejecutar:  ./setup_venv.sh
# ============================================================

set -e

echo "============================================================"
echo "  Creando entorno virtual para Ball-on-Plate RL"
echo "============================================================"

# Detectar Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: No se encontró Python. Instala Python 3.10+"
    exit 1
fi

echo "Usando: $($PYTHON --version)"

# Crear venv
if [ -d "venv" ]; then
    echo "El directorio 'venv' ya existe. ¿Deseas recrearlo? (s/N)"
    read -r answer
    if [[ "$answer" =~ ^[Ss]$ ]]; then
        rm -rf venv
        $PYTHON -m venv venv
    fi
else
    $PYTHON -m venv venv
fi

# Activar
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
echo ""
echo "Instalando dependencias (puede tardar unos minutos)..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "  ¡Entorno listo!"
echo "============================================================"
echo ""
echo "Para activar el entorno en el futuro:"
echo "    source venv/bin/activate"
echo ""
echo "Para entrenar:"
echo "    python scripts/train_sac.py"
echo ""
echo "Para probar un modelo:"
echo "    python scripts/test_agent.py"
echo ""
echo "Para ver TensorBoard:"
echo "    tensorboard --logdir logs/tb"
echo "============================================================"
