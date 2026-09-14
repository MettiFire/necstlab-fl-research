#!/bin/bash
# Fix script: copia i moduli custom nel workspace dopo prepare-jobs-dir
# Uso: ./fix_custom_deploy.sh

set -e

WORKSPACE_BASE="$HOME/fl_benchmark/results/nvflare_poc_workspace"
JOB_SOURCE="$HOME/fl_benchmark/benchmarks/nvidia_flare_poc/jobs/xgb_fedavg_poc"

echo "[*] Cercando app directories nel workspace..."

# Copia custom nel server app
find "$WORKSPACE_BASE" -type d -name "app_server" | while read app_dir; do
    echo "[+] Aggiungendo custom/ in: $app_dir"
    mkdir -p "$app_dir/custom"
    cp -v "$JOB_SOURCE/server_app/custom/__init__.py" "$app_dir/custom/" 2>/dev/null || true
    cp -v "$JOB_SOURCE/server_app/custom/server_controller.py" "$app_dir/custom/" 2>/dev/null || true
    cp -v "$JOB_SOURCE/server_app/custom/client_executor.py" "$app_dir/custom/" 2>/dev/null || true
done

# Copia custom nei client app (app_site-N)
find "$WORKSPACE_BASE" -type d -name "app_site-*" | while read app_dir; do
    echo "[+] Aggiungendo custom/ in: $app_dir"
    mkdir -p "$app_dir/custom"
    cp -v "$JOB_SOURCE/client_app/custom/__init__.py" "$app_dir/custom/" 2>/dev/null || true
    cp -v "$JOB_SOURCE/client_app/custom/client_executor.py" "$app_dir/custom/" 2>/dev/null || true
    cp -v "$JOB_SOURCE/client_app/custom/server_controller.py" "$app_dir/custom/" 2>/dev/null || true
done

echo "[✓] Fix completato! I moduli custom sono stati copiati nel workspace."
