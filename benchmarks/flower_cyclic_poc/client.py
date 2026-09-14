"""
Flower Cyclic Benchmark - Client (PoC legacy-compatible)
NECSTLab - Polimi LS2

Client Flower compatibile con API legacy (`start_client` + `NumPyClient`) e
con logging round-level per tempi/bytes in results/structured_metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
import warnings
from pathlib import Path

import flwr as fl
import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import DataLoader, append_client_round_metric

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# TRACCIAMENTO TIMING GLOBALE (TIMESTAMP ASSOLUTI)
# ============================================================================
INIZIO_TIMING_GLOBALE = None
EVENTI_TIMING = {}  # Traccia tutti gli eventi con timestamp assoluti (epoch UNIX)

def inizializza_timing():
    """Inizializza il timing assoluto all'avvio del client."""
    global INIZIO_TIMING_GLOBALE, EVENTI_TIMING
    INIZIO_TIMING_GLOBALE = time.time()
    EVENTI_TIMING = {}


def ripulisci_file_timing_temporanei():
    """Rimuove i file di timing precedenti per evitare aggregazioni di run vecchi."""
    for file_path in Path("/tmp").glob("timing_client_*.json"):
        try:
            file_path.unlink()
        except Exception:
            pass

def registra_timing(numero_round: int, id_client: int, evento: str, dettagli: str = ""):
    """Registra un evento di timing con timestamp assoluto (epoca UNIX in secondi)."""
    timestamp_assoluto = time.time()
    chiave = f"R{numero_round}_C{id_client}_{evento}"
    EVENTI_TIMING[chiave] = {
        'timestamp': timestamp_assoluto,  # Timestamp assoluto (epoch)
        'details': dettagli
    }
    # Log minimalista per debug
    elapsed_locale = (timestamp_assoluto - INIZIO_TIMING_GLOBALE) * 1000
    if int(elapsed_locale) % 5000 < 100:
        print(f"[TIMING] {int(elapsed_locale/1000)}s elapsed", flush=True)

def stampa_riepilogo_timing():
    """Stampa un riassunto finale con calcoli delle fasi e comunicazione."""
    if not EVENTI_TIMING:
        return

    import json
    import glob
    
    # === STEP 1: Estrai l'ID del client attuale ===
    id_client_attuale = None
    for chiave in EVENTI_TIMING.keys():
        parti = chiave.split('_')
        if len(parti) >= 2 and parti[1].startswith('C'):
            id_client_attuale = int(parti[1][1:])
            break
    
    # === STEP 2: Tutti i client salvano i loro dati in file ===
    if id_client_attuale is not None:
        import os
        pid = os.getpid()
        percorso_timing = f'/tmp/timing_client_{id_client_attuale}_pid{pid}.json'
        try:
            with open(percorso_timing, 'w') as f:
                json.dump(EVENTI_TIMING, f, indent=2)
        except Exception:
            pass  # Ignora errori di file
    
    # === STEP 3: Solo il client 0 aggrega e stampa ===
    if id_client_attuale is not None and id_client_attuale != 0:
        return
    
    # === STEP 4: Client 0 legge tutti i file e li aggrega ===
    dati_aggregati = {}
    try:
        for file_path in sorted(glob.glob('/tmp/timing_client_*.json')) + sorted(glob.glob('/tmp/timing_client_*pid*.json')):

            try:
                with open(file_path, 'r') as f:
                    dati_temp = json.load(f)
                    dati_aggregati.update(dati_temp)
            except Exception:
                pass  # Ignora file non leggibili
    except Exception:
        dati_aggregati = EVENTI_TIMING  # Fallback ai dati locali
    
    if not dati_aggregati:
        dati_aggregati = EVENTI_TIMING
    
    # === STEP 5: Raggruppa per round (filtra round 0) ===
    riepilogo_round = {}
    for chiave, dati in sorted(dati_aggregati.items()):
        parti = chiave.split('_')
        if len(parti) < 3:
            continue
        
        numero_round = int(parti[0][1:])
        id_client = int(parti[1][1:])
        evento = '_'.join(parti[2:])
        
        # FILTRO: Ignora il round 0 (avvio client)
        if numero_round == 0:
            continue
        
        if numero_round not in riepilogo_round:
            riepilogo_round[numero_round] = {}
        if id_client not in riepilogo_round[numero_round]:
            riepilogo_round[numero_round][id_client] = {}
        
        riepilogo_round[numero_round][id_client][evento] = dati['timestamp']
    
    if not riepilogo_round:
        return
    
    # === STEP 5.5: Carica timestamp server per calcolare latenze ===
    timestamp_server_per_round = {}  # {round_num: {t1_send, t10_recv}}

    def _timestamp_server_to_epoch(evento: dict) -> float | None:
        if 'timestamp_epoch' in evento:
            try:
                return float(evento['timestamp_epoch'])
            except Exception:
                return None
        timestamp = evento.get('timestamp')
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        if isinstance(timestamp, str):
            try:
                parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except Exception:
                return None
        return None

    try:
        server_profile_path = Path(__file__).parent / "results" / "server_round_profile.jsonl"



        if server_profile_path.exists():
            with open(server_profile_path, 'r') as f:
                for linea in f:
                    try:
                        evento = json.loads(linea)
                        timestamp_server = _timestamp_server_to_epoch(evento)
                        if timestamp_server is None:
                            continue
                        if evento.get('event') == 'configure_fit':
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t1_send'] = timestamp_server
                        elif evento.get('event') == 'aggregate_fit':
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t10_recv'] = timestamp_server
                    except Exception:
                        pass
        if timestamp_server_per_round:
            print(f"[SUCCESS] Caricati timestamp server per {len(timestamp_server_per_round)} round", flush=True)
    except Exception as exc:
        print(f"[WARN] Errore lettura timestamp server: {exc}", flush=True)
    
        # DEBUG
    print(f"[DEBUG] riepilogo_round contiene {len(riepilogo_round)} round", flush=True)
    print(f"[DEBUG] Round nell'aggregato: {sorted(riepilogo_round.keys())}", flush=True)
    

    larghezza = 110
    separatore = "=" * larghezza
    separatore_minore = "-" * larghezza

    print(f"\n{separatore}")
    print(f"{'TIMING SUMMARY - CYCLIC CLIENT (TIMESTAMP ASSOLUTI)':^{larghezza}}")
    print(separatore)
    
    intestazione = (
        f"{'R':>2}  {'C':>2}  {'Recv(ms)':>8}  {'Deserial':>9}  {'Training':>9}  "
        f"{'Serialize':>9}  {'Fit(ms)':>8}  {'LatenzaUp':>9}  {'LatenzaDn':>9}"
    )
    print()
    print(intestazione)
    print(separatore_minore)

    for numero_round in sorted(riepilogo_round.keys()):
        dati_round = riepilogo_round[numero_round]
        for id_client in sorted(dati_round.keys()):
            eventi = dati_round[id_client]

            # Timestamp assoluti in secondi (epoch)
            t2_recv = eventi.get('t2_entry', 0)  # Client riceve
            t3_deser = eventi.get('t3_model_deserialized', 0)
            t4_data = eventi.get('t4_data_loaded', 0)
            t5_train = eventi.get('t5_training_start', 0)
            t6_train_end = eventi.get('t6_training_end', 0)
            t7_serial = eventi.get('t7_serialization_start', 0)
            t8_serial_end = eventi.get('t8_serialization_end', 0)
            t9_exit = eventi.get('t9_fit_exit', 0)  # Client esce da fit

            # Calcola durate interne (in ms), usando differenze di timestamp
            tempi_ms = lambda ts1, ts2: (ts2 - ts1) * 1000 if ts2 > ts1 else 0
            
            recv_to_data = tempi_ms(t2_recv, t4_data)  # Dalla ricezione al caricamento dati
            deserial_dur = tempi_ms(t4_data, t3_deser) if t3_deser > t4_data else 0  # Deserializzazione
            training_dur = tempi_ms(t5_train, t6_train_end)  # Training
            serial_dur = tempi_ms(t7_serial, t8_serial_end)  # Serializzazione
            total_fit = tempi_ms(t2_recv, t9_exit)  # Tempo totale fit
            
            # Calcola latenze usando timestamp server se disponibili
            latenza_up = 0.0
            latenza_down = 0.0
            if numero_round in timestamp_server_per_round:
                server_data = timestamp_server_per_round[numero_round]
                if 't1_send' in server_data and t2_recv > 0:
                    latenza_up = (t2_recv - server_data['t1_send']) * 1000  # ms
                if 't10_recv' in server_data and t9_exit > 0:
                    latenza_down = (server_data['t10_recv'] - t9_exit) * 1000  # ms

            print(
                f"{numero_round:>2}  {id_client:>2}  {recv_to_data:>8.0f}  {deserial_dur:>9.0f}  "
                f"{training_dur:>9.0f}  {serial_dur:>9.0f}  {total_fit:>8.0f}  "
                f"{latenza_up:>9.0f}  {latenza_down:>9.0f}"
            )

    # === STEP 7: Statistiche globali ===
    tutti_t2 = [dati['timestamp'] for chiave, dati in dati_aggregati.items() if 't2_entry' in chiave]
    tutti_t9 = [dati['timestamp'] for chiave, dati in dati_aggregati.items() if 't9_fit_exit' in chiave]

    print(separatore_minore)
    print()
    print("STATISTICHE GLOBALI")
    print(separatore_minore)
    
    if tutti_t2 and tutti_t9:
        durata_globale_ms = (max(tutti_t9) - min(tutti_t2)) * 1000
        print(f"  {'Timeline client (min-max)':<30}: {durata_globale_ms:.0f}ms = {durata_globale_ms/1000:.2f}s")
    print(f"  {'Numero round completati':<30}: {len(riepilogo_round)}")
    # num_client_round_1 = len(riepilogo_round[1]) if 1 in riepilogo_round else 0
    # print(f"  {'Numero client per round':<30}: {num_client_round_1}")

    # Trova il primo round con dati (per Cyclic, non tutti i round hanno dati)
    primo_round = min(riepilogo_round.keys()) if riepilogo_round else 1
    num_client_round = len(riepilogo_round[primo_round]) if primo_round in riepilogo_round else 0
    print(f"  {'Numero client nel primo round':<30}: {num_client_round}")
    print(f"  {'Numero round totali con dati':<30}: {len(riepilogo_round)}")

    # Calcola comunicazione media (deserialize + serialize)
    tempi_deserializzazione = []
    tempi_serializzazione = []
    tempi_training = []

    for numero_round in sorted(riepilogo_round.keys()):
        for id_client in sorted(riepilogo_round[numero_round].keys()):
            eventi = riepilogo_round[numero_round][id_client]

            t3_deser = eventi.get('t3_model_deserialized', 0)
            t4_data = eventi.get('t4_data_loaded', 0)
            t5_train = eventi.get('t5_training_start', 0)
            t6_train_end = eventi.get('t6_training_end', 0)
            t7_serial = eventi.get('t7_serialization_start', 0)
            t8_serial_end = eventi.get('t8_serialization_end', 0)

            if t3_deser > t4_data:
                tempi_deserializzazione.append((t3_deser - t4_data) * 1000)
            if t8_serial_end > t7_serial:
                tempi_serializzazione.append((t8_serial_end - t7_serial) * 1000)
            if t5_train and t6_train_end:
                tempi_training.append((t6_train_end - t5_train) * 1000)

    if tempi_deserializzazione:
        print(f"  {'Deserialize (andata)':<30}: Min {min(tempi_deserializzazione):.1f}ms | Max {max(tempi_deserializzazione):.1f}ms | Avg {sum(tempi_deserializzazione)/len(tempi_deserializzazione):.1f}ms")

    if tempi_serializzazione:
        print(f"  {'Serialize (ritorno)':<30}: Min {min(tempi_serializzazione):.1f}ms | Max {max(tempi_serializzazione):.1f}ms | Avg {sum(tempi_serializzazione)/len(tempi_serializzazione):.1f}ms")

    if tempi_training:
        print(f"  {'Training':<30}: Min {min(tempi_training):.1f}ms | Max {max(tempi_training):.1f}ms | Avg {sum(tempi_training)/len(tempi_training):.1f}ms")

    print(f"\n{separatore}\n")

    # === NOTA: Latenze ora calcolate dai timestamp server sincronizzati ===
    if timestamp_server_per_round:
        print("[SUCCESS] Latenze calcolate da timestamp server-client sincronizzati (epoch UNIX).")
    else:
        print("[WARN] Latenze mostrano 0.0 - server timing non disponibile.")
    print()

# Utility di serializzazione del booster XGBoost in JSON UTF-8.
def _serialize_booster_json(bst: xgb.Booster) -> bytes:
    """Serializza il booster in JSON UTF-8 compatibile con strategie FedXgb."""
    try:
        raw = bst.save_raw(raw_format="json")
        json.loads(bytes(raw).decode("utf-8"))
        return bytes(raw)
    except Exception:
        # Fallback robusto cross-version XGBoost: salva/riapri JSON da file.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name
            bst.save_model(tmp_path)
            with open(tmp_path, "rb") as f:
                raw = f.read()
            json.loads(raw.decode("utf-8"))
            return raw
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# Utility di training locale: aggiorna il booster per N round e misura il tempo.
def _local_boost(
    booster_locale: xgb.Booster,
    numero_round_locali: int,
    matrice_train: xgb.DMatrix,
) -> tuple[xgb.Booster, float]:
    """Esegue boosting locale e ritorna il booster aggiornato con il tempo speso."""
    inizio_aggiornamento = time.perf_counter()
    for _ in range(numero_round_locali):
        booster_locale.update(matrice_train, booster_locale.num_boosted_rounds())
    tempo_aggiornamento = time.perf_counter() - inizio_aggiornamento
    return booster_locale, tempo_aggiornamento


# Client Flower cyclic con API legacy (`start_client` + `NumPyClient`).
class XgbCyclicClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        local_epochs: int,
        test_fraction: float,
        objective: str,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
    ) -> None:
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.test_fraction = test_fraction
        self.parametri_base = {
            "objective": objective,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        
        # === CACHE DATI: Carica una sola volta all'avvio del client ===
        radice_progetto = Path(__file__).parent.parent.parent
        percorso_dati = str(radice_progetto / "data" / "ml_ready_final_fed")
        self.caricatore_dati = DataLoader(data_dir=percorso_dati)
        self.matrice_train, self.matrice_validazione, self.numero_campioni_train, self.numero_campioni_validazione = \
            self.caricatore_dati.load_client_data(
                client_id=self.client_id,
                test_fraction=self.test_fraction,
            )

    def get_parameters(self, config):
        _ = config
        return []

    def fit(self, parameters, config):
        # Legge la configurazione ricevuta dal server.
        inizio_totale = time.time()
        registra_timing(int(config.get("server-round", 1)), self.client_id, "t2_entry", "")
        
        inizio_configurazione = time.perf_counter()

        round_server = int(config.get("server-round", 1))
        epoche_locali = int(config.get("local-epochs", self.local_epochs))
        frazione_test = float(config.get("test-fraction", self.test_fraction))

        parametri_xgb = {
            "objective": str(config.get("objective", self.parametri_base["objective"])),
            "max_depth": int(config.get("max-depth", self.parametri_base["max_depth"])),
            "learning_rate": float(config.get("learning-rate", self.parametri_base["learning_rate"])),
            "subsample": float(config.get("subsample", self.parametri_base["subsample"])),
            "colsample_bytree": float(config.get("colsample-bytree", self.parametri_base["colsample_bytree"])),
            "nthread": 1,
        }
        tempo_configurazione = time.perf_counter() - inizio_configurazione

        # === CACHE DATI: Usa i dati cachati dal __init__() ===
        inizio_caricatore = time.perf_counter()
        tempo_inizializzazione_caricatore = time.perf_counter() - inizio_caricatore

        inizio_caricamento_dati = time.perf_counter()
        matrice_train = self.matrice_train
        numero_campioni_train = self.numero_campioni_train
        tempo_caricamento_dati = time.perf_counter() - inizio_caricamento_dati
        registra_timing(round_server, self.client_id, "t4_data_loaded", "")
        tempo_caricamento_totale = time.time() - inizio_totale

        byte_modello_in_input = int(parameters[0].nbytes) if parameters else 0
        tempo_deserializzazione = 0.0
        id_esecuzione = str(config.get("run-id", "legacy"))
        round_booster_iniziali = 0
        round_booster_finali = 0
        tempo_aggiornamento_locale = 0.0

        # Esegue il training: primo round da zero, poi aggiornamento incrementale.
        registra_timing(round_server, self.client_id, "t5_training_start", "")
        inizio_training = time.time()
        if round_server == 1 or not parameters or parameters[0].size == 0:
            ramo_training = "cold_start_train"
            booster = xgb.train(parametri_xgb, matrice_train, num_boost_round=epoche_locali)
            round_booster_finali = int(booster.num_boosted_rounds())
        else:
            ramo_training = "incremental_boost"
            inizio_deserializzazione = time.time()
            booster = xgb.Booster(params=parametri_xgb)
            booster.load_model(bytearray(parameters[0].tobytes()))
            tempo_deserializzazione = time.time() - inizio_deserializzazione
            registra_timing(round_server, self.client_id, "t3_model_deserialized", "")

            round_booster_iniziali = int(booster.num_boosted_rounds())
            booster, tempo_aggiornamento_locale = _local_boost(
                booster,
                epoche_locali,
                matrice_train,
            )
            round_booster_finali = int(booster.num_boosted_rounds())
        tempo_training = time.time() - inizio_training
        registra_timing(round_server, self.client_id, "t6_training_end", "")

        # Serializza il modello aggiornato per inviarlo al server.
        registra_timing(round_server, self.client_id, "t7_serialization_start", "")
        inizio_serializzazione = time.time()
        modello_serializzato = _serialize_booster_json(booster)
        vettore_modello = np.frombuffer(modello_serializzato, dtype=np.uint8)
        byte_modello_in_output = int(vettore_modello.nbytes)
        tempo_serializzazione = time.time() - inizio_serializzazione
        registra_timing(round_server, self.client_id, "t8_serialization_end", "")

        righe_matrice = int(matrice_train.num_row())
        colonne_matrice = int(matrice_train.num_col())
        tempo_totale = time.time() - inizio_totale

        metrics = {
            "num-examples": int(numero_campioni_train),
            "tempo_training_secondi": float(tempo_training),
            "tempo_caricamento_secondi": float(tempo_caricamento_totale),
            "tempo_deserializzazione_secondi": float(tempo_deserializzazione),
            "tempo_serializzazione_secondi": float(tempo_serializzazione),
            "tempo_comunicazione_secondi": float(tempo_deserializzazione + tempo_serializzazione),
            "byte_ricevuti": int(byte_modello_in_input),
            "byte_inviati": int(byte_modello_in_output),
            "tempo_totale_secondi": float(tempo_totale),
            # Metriche di profiling per capire dove si spende tempo nel client.
            "ramo_training": ramo_training,
            "tempo_parsing_config_secondi": float(tempo_configurazione),
            "tempo_creazione_dataloader_secondi": float(tempo_inizializzazione_caricatore),
            "tempo_caricamento_dati_secondi": float(tempo_caricamento_dati),
            "tempo_aggiornamento_booster_secondi": float(tempo_aggiornamento_locale),
            "tempo_slicing_bagging_secondi": 0.0,
            "round_booster_prima_training": int(round_booster_iniziali),
            "round_booster_dopo_training": int(round_booster_finali),
            "delta_round_booster": int(round_booster_finali - round_booster_iniziali),
            "righe_matrice_training": int(righe_matrice),
            "colonne_matrice_training": int(colonne_matrice),
            "round_boosting_locali_richiesti": int(epoche_locali),
        }

        append_client_round_metric(
            approach="flower_cyclic",
            client_id=self.client_id,
            round_number=round_server,
            run_id=id_esecuzione,
            metric_row=metrics,
        )

        # t9: Client rimanda i dati al server (esce da fit)
        registra_timing(round_server, self.client_id, "t9_fit_exit", f"Timestamp uscita: {time.time()}")
        return [vettore_modello], int(numero_campioni_train), metrics

    def evaluate(self, parameters, config):
        # MAE temporaneamente disattivata: lasciamo solo il flusso timing/training.
        # Manteniamo il metodo per compatibilita' con NumPyClient.
        _ = parameters
        _ = config
        numero_campioni_validazione = self.numero_campioni_validazione
        return float("inf"), int(numero_campioni_validazione), {
            "mae_disabled": 1.0,
            "num-examples": int(numero_campioni_validazione),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avvia client Flower Cyclic PoC (legacy)")
    parser.add_argument("--server_address", type=str, required=True, help="Es: 127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True, help="ID numerico client")
    parser.add_argument("--local_epochs", type=int, default=1, help="Default local epochs")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Validation split")
    parser.add_argument("--objective", type=str, default="reg:squarederror", help="Objective XGBoost")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth XGBoost")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate XGBoost")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample XGBoost")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Colsample bytree XGBoost")
    args = parser.parse_args()

    ripulisci_file_timing_temporanei()

    # Inizializza il timing globale
    inizializza_timing()
    registra_timing(0, args.client_id, "avvio_client", f"Client {args.client_id} avviato")

    client = XgbCyclicClient(
        client_id=args.client_id,
        local_epochs=args.local_epochs,
        test_fraction=args.test_fraction,
        objective=args.objective,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client(),
        )
    finally:
        
        # Stampa il summary finale quando il client termina
        stampa_riepilogo_timing()


if __name__ == "__main__":
    main()
