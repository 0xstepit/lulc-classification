import dataclasses
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from src.config.reporter import DatetimeConfig, JSONConfig
from src.reporter.models import ReportEntry

SUFFIX = ".json"


def _json(obj):
    # Since is_dataclass is True for both the class and the instance, and a class is also type, we
    # remove the class with the second condition. This way, we check only instances.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.tolist()
    raise TypeError(f"cannot serialize type {type(obj).__name__} to JSON")


class JSONWriter:
    def __init__(
        self, output_dir: Path, file_config: JSONConfig, dt_config: DatetimeConfig
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.indent = file_config.indent
        self.dt_format = dt_config.format

    def write(
        self, entries: list[ReportEntry], filename: str, timestamped: bool = False
    ) -> Path:
        filename_path = Path(filename)

        if SUFFIX not in filename_path.suffix:
            raise ValueError(
                f"filename should be a JSON file, obtained {filename_path.suffix}"
            )

        if timestamped:
            filename = f"{filename_path.stem}_{datetime.now(timezone.utc).strftime(self.dt_format)}{SUFFIX}"

        path = self.output_dir / filename
        payload = [
            {
                "step": e.step,
                "timestamp": e.timestamp,
                "data": e.data,
                "metadata": e.metadata,
            }
            for e in entries
        ]

        path.write_text(json.dumps(payload, indent=self.indent, default=_json))

        return path
