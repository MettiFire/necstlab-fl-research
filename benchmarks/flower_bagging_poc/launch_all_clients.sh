#!/bin/bash
# Script per avviare tutti i client Flower in parallelo (modalità PoC)
# Usage: ./launch_all_clients.sh <NUM_CLIENTS> <SERVER_ADDRESS>

NUM_CLIENTS=${1:-9}  # Default 9 client se non specificato
SERVER_ADDRESS=${2:-127.0.0.1:8080}

for ((i=0; i<$NUM_CLIENTS; i++))
do
  echo "Avvio client $i"
  python client.py --server_address=$SERVER_ADDRESS --client_id=$i &
done

echo "Tutti i client sono stati avviati in background."
wait
