from pathlib import Path
from typing import Any

from lulc.config import ReporterConfig

from .models import ReportEntry
from .writers import JSONWriter

WRITER_REGISTRY: dict[str, type] = {"json": JSONWriter}


class Reporter:
    def __init__(self, output_dir: Path, cfg: ReporterConfig):
        self.output_dir = output_dir
        self._entries: list[ReportEntry] = []

        writer_cls = WRITER_REGISTRY.get(cfg.writer)
        if writer_cls is None:
            raise ValueError(f"unsupported writer: {cfg.writer}")

        self._writer = writer_cls(output_dir, cfg.file_config, cfg.dt)

    def add(self, step: str, data: Any, **metadata) -> None:
        self._entries.append(ReportEntry(step=step, data=data, metadata=metadata))

    def save(self, filename: str) -> Path:
        return self._writer.write(self._entries, filename)
