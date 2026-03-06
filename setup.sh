#!/bin/bash
# Setup script per FL Benchmark Project
# NECSTLab - Polimi LS2

echo "🚀 Setup FL Benchmark Environment"
echo "=================================="
echo ""

# Directory base
BENCHMARK_DIR="/Users/annamettifogo/Desktop/polimi/necstlab/progetto LS2/fl_benchmark"
DATA_SOURCE="/Users/annamettifogo/Desktop/polimi/1° magistrale/csi/proj4/prova1/dtbagging"

cd "$BENCHMARK_DIR"

# 1. Crea virtual environment
echo "📦 Creazione virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Installa dipendenze
echo ""
echo "📚 Installazione dipendenze..."
pip install --upgrade pip
pip install -e .

# 3. Setup symlink dati
echo ""
echo "🔗 Setup symlink ai dati Garmin..."

if [ ! -d "data/ready_for_flwr" ]; then
    ln -s "$DATA_SOURCE/ready_for_flwr" data/ready_for_flwr
    echo "   ✅ Symlink ready_for_flwr creato"
else
    echo "   ℹ️  Symlink ready_for_flwr già esistente"
fi

if [ ! -f "data/x_test.csv" ]; then
    ln -s "$DATA_SOURCE/x_test.csv" data/x_test.csv
    echo "   ✅ Symlink x_test.csv creato"
else
    echo "   ℹ️  Symlink x_test.csv già esistente"
fi

# 4. Test data loading (usa: python test_data.py)
# echo ""
# echo "🧪 Test caricamento dati..."
# python utils.py  # Rimosso: troppo lento, usa test_data.py invece

# 5. Crea directory risultati
mkdir -p results/plots
mkdir -p benchmarks/flower_bagging/results
mkdir -p benchmarks/flower_cyclic/results
mkdir -p benchmarks/nvidia_flare/results

echo ""
echo "=================================="
echo "✅ Setup completato!"
echo ""
echo "📝 Next steps:"
echo "   1. Attiva environment: source venv/bin/activate"
echo "   2. Test Flower bagging: cd benchmarks/flower_bagging && python run_benchmark.py"
echo "   3. Leggi README.md per info complete"
echo ""
