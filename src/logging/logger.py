# src/logging/logger.py

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class JSONLLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_dir()


    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)


    def log(self, record: Dict[str, Any], flush: bool = True):
        #Append a single experiment record to the JSONL file.
        enriched = self._enrich(record)

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched) + "\n")
            if flush:
                f.flush()


    def log_batch(self, records: list[Dict[str, Any]]):
        #Log multiple records efficiently.
        with open(self.filepath, "a", encoding="utf-8") as f:
            for record in records:
                enriched = self._enrich(record)
                f.write(json.dumps(enriched) + "\n")
            f.flush()


    def _enrich(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # Adds metadata to every logged entry.
        enriched = dict(record)
        enriched["timestamp"] = datetime.now().isoformat()
        return enriched
    

    def contains(self, run_id: str) -> bool:
        # Check if a run_id already exists in the log.
        if not os.path.exists(self.filepath):
            return False
        with open(self.filepath, "r", encoding="utf-8") as f:
            return any(
                json.loads(line).get("run_id") == run_id
                for line in f
            )


def log_jsonl(filepath: str, record: Dict[str, Any]):
    logger = JSONLLogger(filepath)
    logger.log(record)
    