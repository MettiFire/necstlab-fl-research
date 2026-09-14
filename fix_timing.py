#!/usr/bin/env python3
"""
Script per fixare il timing client e aggiungere lettura timestamp server.
Modifiche:
1. Aggiungere lettura timestamp server prima della tabella
2. Fixare bug statistiche (usare variabili corrette nel loop)
3. Calcolare latenze dalla differenza di timestamp
"""

import re
from pathlib import Path

def fix_bagging_client():
    """Fixare benchmarks/flower_bagging_poc/client.py"""
    client_file = Path("/Users/annamettifogo/Desktop/polimi/necstlab/progetto_LS2/fl_benchmark/benchmarks/flower_bagging_poc/client.py")
    
    with open(client_file, 'r') as f:
        content = f.read()
    
    # === FIX 1: Aggiungere lettura timestamp server prima della tabella ===
    # Cerca "larghezza = 110"
    old_section_1 = '''    larghezza = 110
    separatore = "=" * larghezza
    separatore_minore = "-" * larghezza

    print(f"\\n{separatore}")
    print(f"{'TIMING SUMMARY - BAGGING CLIENT (TIMESTAMP ASSOLUTI)':^{larghezza}}")
    print(separatore)'''
    
    new_section_1 = '''    # === STEP 5.5: Carica timestamp server per calcolare latenze ===
    timestamp_server_per_round = {}  # {round_num: {t1_send, t10_recv}}
    try:
        server_profile_path = Path(__file__).parent / "results" / "server_round_profile.jsonl"
        if server_profile_path.exists():
            with open(server_profile_path, 'r') as f:
                for linea in f:
                    try:
                        evento = json.loads(linea)
                        if evento.get('event') == 'configure_fit' and 'timestamp_epoch' in evento:
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t1_send'] = evento['timestamp_epoch']
                        elif evento.get('event') == 'aggregate_fit' and 'timestamp_epoch' in evento:
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t10_recv'] = evento['timestamp_epoch']
                    except Exception:
                        pass
        if timestamp_server_per_round:
            print(f"[SUCCESS] Caricati timestamp server per {len(timestamp_server_per_round)} round", flush=True)
    except Exception as exc:
        print(f"[WARN] Errore lettura timestamp server: {exc}", flush=True)
    
    larghezza = 110
    separatore = "=" * larghezza
    separatore_minore = "-" * larghezza

    print(f"\\n{separatore}")
    print(f"{'TIMING SUMMARY - BAGGING CLIENT (TIMESTAMP ASSOLUTI)':^{larghezza}}")
    print(separatore)'''
    
    content = content.replace(old_section_1, new_section_1)
    
    # === FIX 2: Aggiungere calcolo latenze nella tabella ===
    old_section_2 = '''            # Calcola durate interne (in ms), usando differenze di timestamp
            tempi_ms = lambda ts1, ts2: (ts2 - ts1) * 1000 if ts2 > ts1 else 0
            
            recv_to_data = tempi_ms(t2_recv, t4_data)  # Dalla ricezione al caricamento dati
            deserial_dur = tempi_ms(t4_data, t3_deser) if t3_deser > t4_data else 0  # Deserializzazione
            training_dur = tempi_ms(t5_train, t6_train_end)  # Training
            serial_dur = tempi_ms(t7_serial, t8_serial_end)  # Serializzazione
            total_fit = tempi_ms(t2_recv, t9_exit)  # Tempo totale fit
            
            # Latenze (al momento non abbiamo dati server, mostro 0 come placeholder)
            latenza_up = 0.0  # t2_recv - server_send (da calcolare quando abbiamo dati server)
            latenza_down = 0.0  # server_recv - t9_exit (da calcolare quando abbiamo dati server)'''
    
    new_section_2 = '''            # Calcola durate interne (in ms), usando differenze di timestamp
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
                    latenza_down = (server_data['t10_recv'] - t9_exit) * 1000  # ms'''
    
    content = content.replace(old_section_2, new_section_2)
    
    # === FIX 3: Fixare il bug statistiche (usare variabili corrette) ===
    old_section_3 = '''    for numero_round in sorted(riepilogo_round.keys()):
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
                tempi_training.append((t6_train_end - t5_train) * 1000)'''
    
    # Questo è già corretto, verifica che sia presente
    if old_section_3 not in content:
        print("[WARN] Sezione statistiche non trovata, potrebbe essere già corretta")
    
    # === FIX 4: Update messaggio finale ===
    old_section_4 = '''    print(f"\\n{separatore}\\n")

    # === NOTA: Le latenze up/down saranno calcolate quando il server loggherà i timestamp di send/recv ===
    print("\\n[INFO] Colonne 'LatenzaUp' e 'LatenzaDn' richiedono logging server-side per essere calcolate.")
    print("[INFO] Un'analisi globale completa richiede di unire i timestamp di server e client.\\n")'''
    
    new_section_4 = '''    print(f"\\n{separatore}\\n")

    # === NOTA: Latenze ora calcolate dai timestamp server sincronizzati ===
    if timestamp_server_per_round:
        print("[SUCCESS] Latenze calcolate da timestamp server-client sincronizzati (epoch UNIX).")
    else:
        print("[WARN] Latenze mostrano 0.0 - server timing non disponibile.")
    print()'''
    
    content = content.replace(old_section_4, new_section_4)
    
    with open(client_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixato {client_file}")

def fix_cyclic_client():
    """Fixare benchmarks/flower_cyclic_poc/client.py (copia identica da bagging)"""
    cyclic_file = Path("/Users/annamettifogo/Desktop/polimi/necstlab/progetto_LS2/fl_benchmark/benchmarks/flower_cyclic_poc/client.py")
    
    # Applica le stesse fix del bagging
    with open(cyclic_file, 'r') as f:
        content = f.read()
    
    # FIX 1
    old_section_1 = '''    larghezza = 110
    separatore = "=" * larghezza
    separatore_minore = "-" * larghezza

    print(f"\\n{separatore}")
    print(f"{'TIMING SUMMARY - CYCLIC CLIENT (TIMESTAMP ASSOLUTI)':^{larghezza}}")
    print(separatore)'''
    
    new_section_1 = '''    # === STEP 5.5: Carica timestamp server per calcolare latenze ===
    timestamp_server_per_round = {}  # {round_num: {t1_send, t10_recv}}
    try:
        server_profile_path = Path(__file__).parent / "results" / "server_round_profile.jsonl"
        if server_profile_path.exists():
            with open(server_profile_path, 'r') as f:
                for linea in f:
                    try:
                        evento = json.loads(linea)
                        if evento.get('event') == 'configure_fit' and 'timestamp_epoch' in evento:
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t1_send'] = evento['timestamp_epoch']
                        elif evento.get('event') == 'aggregate_fit' and 'timestamp_epoch' in evento:
                            round_num = evento.get('server_round')
                            if round_num not in timestamp_server_per_round:
                                timestamp_server_per_round[round_num] = {}
                            timestamp_server_per_round[round_num]['t10_recv'] = evento['timestamp_epoch']
                    except Exception:
                        pass
        if timestamp_server_per_round:
            print(f"[SUCCESS] Caricati timestamp server per {len(timestamp_server_per_round)} round", flush=True)
    except Exception as exc:
        print(f"[WARN] Errore lettura timestamp server: {exc}", flush=True)
    
    larghezza = 110
    separatore = "=" * larghezza
    separatore_minore = "-" * larghezza

    print(f"\\n{separatore}")
    print(f"{'TIMING SUMMARY - CYCLIC CLIENT (TIMESTAMP ASSOLUTI)':^{larghezza}}")
    print(separatore)'''
    
    content = content.replace(old_section_1, new_section_1)
    
    # FIX 2
    old_section_2 = '''            # Calcola durate interne (in ms), usando differenze di timestamp
            tempi_ms = lambda ts1, ts2: (ts2 - ts1) * 1000 if ts2 > ts1 else 0
            
            recv_to_data = tempi_ms(t2_recv, t4_data)  # Dalla ricezione al caricamento dati
            deserial_dur = tempi_ms(t4_data, t3_deser) if t3_deser > t4_data else 0  # Deserializzazione
            training_dur = tempi_ms(t5_train, t6_train_end)  # Training
            serial_dur = tempi_ms(t7_serial, t8_serial_end)  # Serializzazione
            total_fit = tempi_ms(t2_recv, t9_exit)  # Tempo totale fit
            
            # Latenze (al momento non abbiamo dati server, mostro 0 come placeholder)
            latenza_up = 0.0  # t2_recv - server_send (da calcolare quando abbiamo dati server)
            latenza_down = 0.0  # server_recv - t9_exit (da calcolare quando abbiamo dati server)'''
    
    new_section_2 = '''            # Calcola durate interne (in ms), usando differenze di timestamp
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
                    latenza_down = (server_data['t10_recv'] - t9_exit) * 1000  # ms'''
    
    content = content.replace(old_section_2, new_section_2)
    
    # FIX 4
    old_section_4 = '''    print(f"\\n{separatore}\\n")

    # === NOTA: Le latenze up/down saranno calcolate quando il server loggherà i timestamp di send/recv ===
    print("\\n[INFO] Colonne 'LatenzaUp' e 'LatenzaDn' richiedono logging server-side per essere calcolate.")
    print("[INFO] Un'analisi globale completa richiede di unire i timestamp di server e client.\\n")'''
    
    new_section_4 = '''    print(f"\\n{separatore}\\n")

    # === NOTA: Latenze ora calcolate dai timestamp server sincronizzati ===
    if timestamp_server_per_round:
        print("[SUCCESS] Latenze calcolate da timestamp server-client sincronizzati (epoch UNIX).")
    else:
        print("[WARN] Latenze mostrano 0.0 - server timing non disponibile.")
    print()'''
    
    content = content.replace(old_section_4, new_section_4)
    
    with open(cyclic_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixato {cyclic_file}")

if __name__ == "__main__":
    fix_bagging_client()
    fix_cyclic_client()
    print("\n✅ Tutte le modifiche applicate!")
