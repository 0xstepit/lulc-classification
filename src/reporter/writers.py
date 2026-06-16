import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.reporter import DatetimeConfig, JSONConfig
from src.reporter.models import ReportEntry

SUFFIX = ".json"


class JSONWriter:
    def __init__(
        self, output_dir: Path, file_config: JSONConfig, dt_config: DatetimeConfig
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.indent = file_config.indent
        self.dt_format = dt_config.format

    def write(
        self, entries: list[ReportEntry], filename: str, timestamped: bool = True
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
                "timestamp": e.timestamp.isoformat(),
                "data": e.data,
                "metadata": e.metadata,
            }
            for e in entries
        ]

        path.write_text(json.dumps(payload, indent=self.indent, default=str))

        return path
