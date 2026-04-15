import os
import json
import time


class MetricsLogger:
    """Structured JSON-lines logger for timing and routing metrics."""

    def __init__(self, node_ip: str, log_dir: str = "~/logs"):
        self.node_ip = node_ip
        log_dir_expanded = os.path.expanduser(log_dir)
        os.makedirs(log_dir_expanded, exist_ok=True)
        self.log_path = os.path.join(log_dir_expanded, f"metrics_{node_ip}.jsonl")
        self._file = open(self.log_path, "a")

    def log(self, event_type: str, **kwargs):
        record = {
            "ts": time.time(),
            "node": self.node_ip,
            "event": event_type,
            **kwargs,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
