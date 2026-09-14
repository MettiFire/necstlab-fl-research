#!/bin/bash
# Killa tutti i processi NVFlare in modo brutale ma pulito

echo "=== Killing NVFlare processes ==="

# 1. Prova lo stop ufficiale (spesso non basta, ma ci proviamo)
pkill -f "fl_server" 2>/dev/null
pkill -f "fl_client" 2>/dev/null  
pkill -f "admin_client" 2>/dev/null

# 2. Killa per porta (server usa 8002 e 8003)
fuser -k 8002/tcp 2>/dev/null
fuser -k 8003/tcp 2>/dev/null

# 3. Killa tutti i processi python che hanno "nvflare" o "fl_benchmark" nel path
pkill -f "nvflare" 2>/dev/null
pkill -f "fl_benchmark.*python" 2>/dev/null

sleep 2

# 4. Verifica
REMAINING=$(ps aux | grep -E "fl_server|fl_client|nvflare" | grep -v grep)
if [ -z "$REMAINING" ]; then
    echo "✓ Tutti i processi NVFlare terminati"
else
    echo "⚠ Processi ancora in vita, force kill..."
    echo "$REMAINING"
    pkill -9 -f "fl_server" 2>/dev/null
    pkill -9 -f "fl_client" 2>/dev/null
    pkill -9 -f "nvflare" 2>/dev/null
fi

echo "=== Done ==="
