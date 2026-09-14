"""NVFlare Server Controller with timing synchronization.

Sends task to all clients with server_timestamp_t1.
Receives results with client_timestamp_t9 and calculates latenza_down.

FIX: usa Task + broadcast_and_wait dell'API reale NVFlare invece degli stub vuoti.
FIX: task name fisso "train" (non dinamico per round) per matchare il client.
"""

import time
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))


from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.impl.controller import Controller


class XGBServerController(Controller):
    """FedAvg Server Controller with timing."""

    def __init__(self, num_rounds: int = 10):
        super().__init__()
        self.num_rounds = num_rounds
        self.global_model: bytes = b""

        # Raccogliamo le risposte per round tramite callback
        self._round_responses: dict = {}

    def start_controller(self, fl_ctx: FLContext) -> None:
        """Initialize controller."""
        self.log_info(fl_ctx, "XGBServerController started")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        """Cleanup."""
        self.log_info(fl_ctx, "XGBServerController stopped")

    def process_result_of_unknown_task(
        self,
        client,
        task_name: str,
        client_task_id: str,
        result: Shareable,
        fl_ctx: FLContext,
    ) -> None:
        self.log_warning(fl_ctx, f"Received result for unknown task: {task_name}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """Main federated learning loop."""
        self.log_info(fl_ctx, f"Starting {self.num_rounds} rounds of FedAvg")

        for round_num in range(1, self.num_rounds + 1):
            if abort_signal.triggered:
                self.log_info(fl_ctx, "Abort signal received, stopping.")
                return

            self.log_info(fl_ctx, f"\n{'=' * 70}")
            self.log_info(fl_ctx, f"Round {round_num}/{self.num_rounds}")
            self.log_info(fl_ctx, f"{'=' * 70}")

            timestamp_t1_send = time.time()
            self.log_info(fl_ctx, f"[TIMING_SERVER] R{round_num} t1_send={timestamp_t1_send:.6f}")

            task_data = Shareable()
            task_data["round"] = round_num
            task_data["global_model"] = self.global_model
            task_data["server_timestamp_t1"] = timestamp_t1_send

            self._round_responses = {}

            task = Task(
                name="train",
                data=task_data,
                result_received_cb=self._result_received_cb,
            )

            self.broadcast_and_wait(
                task=task,
                min_responses=0,
                wait_time_after_min_received=10,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            timestamp_t10_recv = time.time()
            self.log_info(fl_ctx, f"[TIMING_SERVER] R{round_num} t10_recv={timestamp_t10_recv:.6f}")

            if not self._round_responses:
                self.log_warning(fl_ctx, f"No responses received for round {round_num}")
                continue

            for client_name, response_data in self._round_responses.items():
                if response_data is None:
                    continue
                client_timestamp_t9 = response_data.get("client_timestamp_t9", 0)
                if client_timestamp_t9 > 0:
                    latenza_down_ms = (timestamp_t10_recv - client_timestamp_t9) * 1000
                    self.log_info(
                        fl_ctx,
                        f"[TIMING_SERVER] R{round_num} {client_name} "
                        f"latenza_down={latenza_down_ms:.1f}ms",
                    )

            self.global_model = self._aggregate_models(self._round_responses)
            self.log_info(fl_ctx, f"Round {round_num} aggregation completed "
                          f"({len(self._round_responses)} responses)")

    def _result_received_cb(self, client_task, fl_ctx: FLContext) -> None:
        client_name = client_task.client.name
        result: Shareable = client_task.result

        if result is None:
            self.log_warning(fl_ctx, f"None result from {client_name}")
            return

        if result.get_return_code() != "OK":
            self.log_warning(fl_ctx, f"Non-OK return code from {client_name}: "
                             f"{result.get_return_code()}")

        response_dict = {
            "model": result.get("model", b""),
            "metrics": result.get("metrics", {}),
            "client_timestamp_t9": result.get("client_timestamp_t9", 0),
        }
        self._round_responses[client_name] = response_dict
        self.log_info(fl_ctx, f"[TIMING_SERVER] received result from {client_name}")

    def _aggregate_models(self, responses: dict) -> bytes:
        for response_data in responses.values():
            if response_data and response_data.get("model"):
                return response_data["model"]
        return self.global_model
