
# Flower Bagging PoC

Questa cartella contiene la versione "proof of concept" (PoC) del benchmark Flower Bagging, con server e client separati per simulare una federazione reale.

## Ruolo dei file

- **server.py**: avvia il server federato Flower, coordina i round e aggrega i modelli dei client.
- **client.py**: avvia un client federato Flower, esegue il training locale e invia gli aggiornamenti al server.
- **launch_all_clients.sh**: script bash per avviare automaticamente tutti i client in parallelo.
- **run_benchmark.py**: (opzionale) script per orchestrare benchmark o automazioni aggiuntive.
- **README.md**: questa guida.

## Come eseguire in modalità PoC

### 1. Avviare il server
Sulla macchina che farà da server (può essere il lab o il tuo PC):

```bash
python server.py --server_address=0.0.0.0:8080 --num_rounds=10
```

### 2. Avviare i client
Su una o più macchine (anche la stessa del server), per ogni client lancia:

```bash
python client.py --server_address=IP_DEL_SERVER:8080 --client_id=N
```
Dove:
- `IP_DEL_SERVER` è l'indirizzo IP della macchina dove gira il server (es: 127.0.0.1 se tutto in locale, oppure l'IP del lab se i client sono su altre macchine)
- `N` è l'ID numerico del client (da 0 a NUM_CLIENTS-1)

### 3. Avvio automatico di tutti i client (opzionale)
Se vuoi avviare tutti i client in parallelo sulla stessa macchina:

```bash
chmod +x launch_all_clients.sh
./launch_all_clients.sh 9 IP_DEL_SERVER:8080
```
Sostituisci 9 con il numero di client desiderato.

### Note importanti
- Il server deve essere avviato prima dei client.
- Assicurati che la porta 8080 sia aperta e raggiungibile tra server e client (controlla firewall/rete).
- Puoi lanciare client su macchine diverse, basta che puntino allo stesso indirizzo server.

### Come trovare l'IP del server
Sul server, esegui:
```bash
hostname -I
# oppure
ip addr show
```
Usa l'indirizzo IP mostrato come IP_DEL_SERVER nei comandi dei client.
