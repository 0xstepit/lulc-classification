import dataclasses
from pathlib import Path

import pytest

from lulc.config.viz import Colors, Font, VizConfig, load_viz_config

REPO_ROOT = Path(__file__).resolve().parents[1]

COLORS_TOML = """
[colors]
text      = "white"
facecolor = "black"

[colors.prop_cycle]
0 = '#1B4965'
1 = '#2A9D8F'
2 = '#E9C46A'
"""

FONT_TOML = """
[font.size]
title    = 18
subtitle = 16
legend   = 9
"""

VALID_TOML = COLORS_TOML + FONT_TOML


def _write_toml(tmp_path: Path, content: str = VALID_TOML) -> Path:
    file_path = tmp_path / "viz.toml"
    file_path.write_text(content)
    return file_path


class TestLoadVizConfig:
    def test_returns_each_section_as_its_own_dataclass(self, tmp_path):
        # Regression: the sections used to be handed over as raw dicts, so
        # cfg.colors.text raised AttributeError.
        cfg = load_viz_config(_write_toml(tmp_path))

        assert isinstance(cfg, VizConfig)
        assert isinstance(cfg.colors, Colors)
        assert isinstance(cfg.font, Font)

    def test_parses_the_colors(self, tmp_path):
        cfg = load_viz_config(_write_toml(tmp_path))

        assert cfg.colors.text == "white"
        assert cfg.colors.facecolor == "black"

    def test_parses_the_font_sizes(self, tmp_path):
        cfg = load_viz_config(_write_toml(tmp_path))

        assert cfg.font.size == {"title": 18, "subtitle": 16, "legend": 9}

    def test_converts_the_toml_string_keys_of_the_prop_cycle(self, tmp_path):
        cfg = load_viz_config(_write_toml(tmp_path))

        assert cfg.colors.prop_cycle == {
            0: "#1B4965",
            1: "#2A9D8F",
            2: "#E9C46A",
        }

    def test_raises_when_the_file_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_viz_config(tmp_path / "missing.toml")

    @pytest.mark.parametrize(
        ("missing", "content"), [("colors", FONT_TOML), ("font", COLORS_TOML)]
    )
    def test_raises_when_a_section_is_missing(self, tmp_path, missing, content):
        with pytest.raises(KeyError, match=missing):
            load_viz_config(_write_toml(tmp_path, content))

    def test_raises_when_a_section_has_an_unknown_key(self, tmp_path):
        with_extra = VALID_TOML.replace('text      = "white"', 'texts = "white"')

        with pytest.raises(TypeError):
            load_viz_config(_write_toml(tmp_path, with_extra))

    def test_the_project_configuration_file_loads(self):
        cfg = load_viz_config(REPO_ROOT / "config" / "viz.toml")

        assert isinstance(cfg.colors, Colors)
        assert isinstance(cfg.font, Font)
        assert all(isinstance(key, int) for key in cfg.colors.prop_cycle)


class TestColors:
    def test_converts_the_prop_cycle_keys_to_int(self):
        colors = Colors(
            text="white", facecolor="black", prop_cycle={0: "#1B4965", 1: "#2A9D8F"}
        )

        assert colors.prop_cycle == {0: "#1B4965", 1: "#2A9D8F"}

    def test_leaves_already_integer_keys_untouched(self):
        colors = Colors(text="white", facecolor="black", prop_cycle={0: "#1B4965"})

        assert colors.prop_cycle == {0: "#1B4965"}

    def test_accepts_an_empty_prop_cycle(self):
        colors = Colors(text="white", facecolor="black", prop_cycle={})

        assert colors.prop_cycle == {}

    def test_is_frozen(self):
        colors = Colors(text="white", facecolor="black", prop_cycle={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            colors.text = "black"  # type: ignore[misc]


class TestFont:
    def test_keeps_the_sizes(self):
        font = Font(size={"title": 18})

        assert font.size == {"title": 18}

    def test_is_frozen(self):
        font = Font(size={"title": 18})

        with pytest.raises(dataclasses.FrozenInstanceError):
            font.size = {"title": 20}  # type: ignore[misc]
