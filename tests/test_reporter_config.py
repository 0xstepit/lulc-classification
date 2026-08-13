from pathlib import Path

import pytest

from lulc.config.reporter import (
    DatetimeConfig,
    JSONConfig,
    ReporterConfig,
    load_reporter_config,
)

VALID_TOML = """
writer = "json"

[dt]
format = "%Y%m%d_%H%M%S"

[json]
indent = 4
"""

UNSUPPORTED_WRITER_TOML = """
writer = "yaml"

[dt]
format = "%Y%m%d_%H%M%S"
"""


def _write_toml(tmp_path: Path, content: str) -> Path:
    file_path = tmp_path / "reporter.toml"
    file_path.write_text(content)
    return file_path


class TestLoadReporterConfig:
    def test_loads_valid_config(self, tmp_path):
        file_path = _write_toml(tmp_path, VALID_TOML)

        cfg = load_reporter_config(file_path)

        assert cfg.writer == "json"
        assert cfg.dt == DatetimeConfig(format="%Y%m%d_%H%M%S")
        assert cfg.file_config == JSONConfig(indent=4)

    def test_returns_reporter_config_instance(self, tmp_path):
        file_path = _write_toml(tmp_path, VALID_TOML)

        cfg = load_reporter_config(file_path)

        assert isinstance(cfg, ReporterConfig)
        assert isinstance(cfg.dt, DatetimeConfig)
        assert isinstance(cfg.file_config, JSONConfig)

    def test_raises_when_file_does_not_exist(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.toml"

        with pytest.raises(FileNotFoundError):
            load_reporter_config(missing_path)

    def test_raises_when_writer_unsupported(self, tmp_path):
        file_path = _write_toml(tmp_path, UNSUPPORTED_WRITER_TOML)

        with pytest.raises(ValueError):
            load_reporter_config(file_path)
