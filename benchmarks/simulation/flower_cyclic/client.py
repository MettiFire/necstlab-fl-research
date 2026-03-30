"""
Flower Cyclic Benchmark - Client
NECSTLab - Polimi LS2
"""
import warnings
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys
import time

# Aggiungi root al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from sklearn.metrics import mean_absolute_error

from utils import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    """Boosting locale per il client"""
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    
    if train_method == "bagging": 
        # Per bagging: ritorna solo gli ultimi alberi addestrati
        bst = bst_input[
            bst_input.num_boosted_rounds() - num_local_round :  
            bst_input.num_boosted_rounds()
        ]
    else: 
        # Per cyclic: ritorna tutto il modello
        bst = bst_input
    
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Training locale del client"""
    start_total = time.time()
    
    # Configurazione
    partition_id = context.node_config["partition-id"]
    num_local_round = context.run_config["local-epochs"]
    train_method = context.run_config.get("train-method", "cyclic")
    test_fraction = context.run_config.get("test-fraction", 0.2)
    
    # Parametri XGBoost
    cfg = unflatten_dict(context.run_config)
    params = {
        "objective": cfg.get("objective", "reg:squarederror"),
        "max_depth": cfg.get("max-depth", 6),
        "learning_rate": cfg.get("learning-rate", 0.1),
        "subsample": cfg.get("subsample", 0.8),
        "colsample_bytree": cfg.get("colsample-bytree", 0.8),
    }
    
    # Carica dati
    start_load = time.time()
    loader = DataLoader()
    train_dmatrix, _, num_train, _ = loader.load_client_data(
        client_id=partition_id,
        test_fraction=test_fraction
    )
    load_time = time.time() - start_load
    
    global_round = msg.content["config"]["server-round"]
    
    # Training
    start_train = time.time()
    if global_round == 1:
        # Primo round: training da zero
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Round successivi: continua dal modello globale
        start_deserialize = time.time()
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)
        deserialize_time = time.time() - start_deserialize
        bst = _local_boost(bst, num_local_round, train_dmatrix, train_method)
    train_time = time.time() - start_train
    
    # Serializza modello locale
    start_serialize = time.time()
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)
    serialize_time = time.time() - start_serialize
    
    total_time = time.time() - start_total
    
    # Prepara risposta
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
    """Evaluation locale del client"""
    
    partition_id = context.node_config["partition-id"]
    test_fraction = context.run_config.get("test-fraction", 0.2)
    
    # Parametri XGBoost
    cfg = unflatten_dict(context.run_config)
    params = {
        "objective": cfg.get("objective", "reg:squarederror"),
        "max_depth": cfg.get("max-depth", 6),
        "learning_rate": cfg.get("learning-rate", 0.1),
    }
    
    # Carica dati validation
    loader = DataLoader()
    _, valid_dmatrix, _, num_val = loader.load_client_data(
        client_id=partition_id,
        test_fraction=test_fraction
    )
    
    # Carica modello globale
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    # Predizione e metriche
    y_pred = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()
    mae = mean_absolute_error(y_true, y_pred)
    
    # Prepara risposta
    metrics = {
        "mae": mae,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
