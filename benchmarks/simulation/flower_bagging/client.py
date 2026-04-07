"""
Flower Bagging Benchmark - Client
NECSTLab - Polimi LS2

implemento la logica lato client Flower:
- training locale del modello XGBoost
- serializzazione del modello da inviare al server
- valutazione locale per ottenere la MAE

"""
import warnings
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys
import time

# aggiungo la root del progetto al path per poter importare `utils`
# anche quando Flower esegue il client da contesti diversi.
sys.path.append(str(Path(__file__).parent.parent.parent))

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp # classe base per implementare il client
from flwr.common.config import unflatten_dict # per ricostruire i parametri XGBoost dalla config piatta di Flower
from sklearn.metrics import mean_absolute_error

from utils import DataLoader

warnings.filterwarnings("ignore", category=UserWarning) # XGBoost a volte emette warning su feature non usate, ecc. che possiamo ignorare.
warnings.filterwarnings("ignore", category=DeprecationWarning) # alcune funzioni di XGBoost sono deprecate ma ancora usate internamente, quindi silenzio anche questi warning.

app = ClientApp()

# La logica di training e valutazione è implementata nei metodi @app.train() e @app.evaluate(),
# che flower chiama automaticamente al momento giusto durante la simulazione federata.
def _local_boost(bst_input, num_local_round, train_dmatrix, train_method): # funzione helper per eseguire il boosting locale del client, con logica diversa per bagging e cyclic
    """
    Eseguo il boosting locale del client per `num_local_round` iterazioni.
    - centralizza la logica condivisa tra round successivi al primo
    - rende esplicita la differenza di output tra bagging e cyclic
    """
    for _ in range(num_local_round):
        # aggiorno il booster un albero alla volta usando i dati locali.
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    
    if train_method == "bagging": 
        # in bagging condivido solo il contributo locale di questo round
        # (gli ultimi `num_local_round` alberi), non l'intero modello.
        bst = bst_input[
            bst_input.num_boosted_rounds() - num_local_round :  
            bst_input.num_boosted_rounds()
        ]
    else: 
        # in cyclic mantengo il modello completo aggiornato localmente.
        bst = bst_input
    
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Eseguo un round di training locale.

    Flusso principale:
    1) leggo la configurazione del round
    2) carico i dati del client
    3) alleno da zero (round 1) o continuo dal modello globale
    4) serializzo il modello risultante e invio metriche temporali
    """
    start_total = time.time()
    
    # identifico il client corrente e i parametri federati del round.
    partition_id = context.node_config["partition-id"]
    num_local_round = context.run_config["local-epochs"]
    train_method = context.run_config.get("train-method", "bagging")
    test_fraction = context.run_config.get("test-fraction", 0.2)
    
    # ricostruisco i parametri XGBoost dalla run-config Flower.
    cfg = unflatten_dict(context.run_config)
    params = {
        "objective": cfg.get("objective", "reg:squarederror"), # per confronto equo con Flower Cyclic, uso sempre regressione con errore quadratico
        "max_depth": cfg.get("max-depth", 6), # profondità massima degli alberi, bilancia complessità e overfitting
        "learning_rate": cfg.get("learning-rate", 0.1), # tasso di apprendimento, controlla la velocità di aggiornamento del modello
        "subsample": cfg.get("subsample", 0.8), # frazione di campioni usati per costruire ogni albero, aiuta a ridurre overfitting e aumenta la diversità tra i modelli locali (importante per bagging)
        "colsample_bytree": cfg.get("colsample-bytree", 0.8), # frazione di feature usate per costruire ogni albero, aiuta a ridurre overfitting e aumenta la diversità tra i modelli locali (importante per bagging)
    }
    
    # carico i dati locali in formato DMatrix: è il formato più efficiente per il training con XGBoost.
    start_load = time.time()
    loader = DataLoader()
    train_dmatrix, _, num_train, _ = loader.load_client_data( # carico sia train che valid per comodità, anche se in questo metodo uso solo train
        client_id=partition_id,
        test_fraction=test_fraction
    )
    load_time = time.time() - start_load
    
    global_round = msg.content["config"]["server-round"] 
    
    # leggo il numero di round server corrente per capire se partire da zero
    # o dal modello globale aggregato al round precedente.
    start_train = time.time()
    if global_round == 1:
        # Round 1: non esiste ancora un modello globale utile,
        # quindi alleno da zero sul dataset locale.
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Round > 1: deserializzo il modello globale ricevuto dal server
        # e continuo il boosting in locale.
        start_deserialize = time.time()
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)
        deserialize_time = time.time() - start_deserialize
        bst = _local_boost(bst, num_local_round, train_dmatrix, train_method)
    train_time = time.time() - start_train
    
    # Serializzo il modello in bytes JSON per inviarlo nel payload Flower.
    # Uso uint8 per avere un buffer compatto e semplice da trasportare.
    start_serialize = time.time()
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)
    serialize_time = time.time() - start_serialize
    
    total_time = time.time() - start_total
    
    # Impacchetto modello + metriche temporali del client.
    # Queste metriche servono per analizzare dove spendo tempo localmente.
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": num_train,
        "train_time": train_time,
        "load_time": load_time,
        "serialize_time": serialize_time,
        "total_time": total_time
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Valuto il modello globale sul validation set locale del client.

    Perche' valuto localmente:
    - rispetto il paradigma federato (i dati restano sul client)
    - ottengo una metrica aggregabile lato server (MAE pesata)
    """
    
    partition_id = context.node_config["partition-id"]
    test_fraction = context.run_config.get("test-fraction", 0.2)
    
    # Mantengo i parametri minimi necessari per ricostruire correttamente
    # il booster che ricevo dal server.
    cfg = unflatten_dict(context.run_config)
    params = {
        "objective": cfg.get("objective", "reg:squarederror"),
        "max_depth": cfg.get("max-depth", 6),
        "learning_rate": cfg.get("learning-rate", 0.1),
    }
    
    # Carico solo la parte di validation per la valutazione locale.
    loader = DataLoader()
    _, valid_dmatrix, _, num_val = loader.load_client_data(
        client_id=partition_id,
        test_fraction=test_fraction
    )
    
    # Deserializzo il modello globale ricevuto dal server.
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    # Calcolo la MAE locale: e' la metrica di qualita' usata nel confronto.
    y_pred = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    mae = mean_absolute_error(y_true, y_pred)
    
    # Invio metrica e numero di esempi per permettere aggregazioni pesate.
    metrics = {
        "mae": mae,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
