from datetime import datetime, timezone
from pathlib import Path


class AlertManager:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(self, engine_id: int, confidence: float) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = (
            f"{timestamp} | engine_id={engine_id} | "
            f"confidence={confidence:.4f}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)
