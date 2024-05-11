import sqlite3
from typing import Literal, Dict

import torch


def synchronize_device_and_clear_cache(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    # Always synchronize CPU
    torch.cpu.synchronize()


class DiskCache:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)"
        )

    def __setitem__(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def __getitem__(self, key):
        cursor = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __contains__(self, item):
        cursor = self.conn.execute("SELECT value FROM cache WHERE key = ?", (item,))
        row = cursor.fetchone()
        return row is not None


DTypeStr = Literal["float32", "float16", "bfloat16"]
DeviceStr = Literal["cpu", "cuda", "mps"]
_dtype_map: Dict[DTypeStr, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DEFAULT_PROMPT = "An astronaut riding on a horse."
