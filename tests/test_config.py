import pytest

from src.config import deep_merge, load_config


class TestDeepMerge:
    def test_does_not_mutate_base(self):
        base = {"a": 1, "b": {"c": 2}}
        overrides = {"b": {"c": 3}}
        deep_merge(base, overrides)
        assert base == {"a": 1, "b": {"c": 2}}

    def test_empty_base(self):
        base = {}
        overrides = {"c": 3}
        assert deep_merge(base, overrides) == {"c": 3}

    def test_no_override(self):
        base = {"a": 1, "b": 2}
        overrides = {"c": 3}
        assert deep_merge(base, overrides) == {"a": 1, "b": 2, "c": 3}

    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        overrides = {"b": 3}
        assert deep_merge(base, overrides) == {"a": 1, "b": 3}

    def test_nested_ovverried(self):
        base = {"a": 1, "b": {"c": 2}}
        overrides = {"b": {"c": 3}}
        assert deep_merge(base, overrides) == {"a": 1, "b": {"c": 3}}


class TestLoadConfig:
    def test_load_single_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)

        config_one = tmp_path / "config_one.toml"
        config_one.write_text("[training]\nepochs=100\n")

        loaded_config = load_config(["config_one.toml"])

        assert loaded_config["training"]["epochs"] == 100

    def test_load_multiple_configs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)

        config_one = tmp_path / "config_one.toml"
        config_one.write_text("[training]\nepochs=100\n")

        config_two = tmp_path / "config_two.toml"
        config_two.write_text("[model]\nin_channels=3\n")

        loaded_config = load_config(["config_one.toml", "config_two.toml"])

        assert loaded_config["training"]["epochs"] == 100
        assert loaded_config["model"]["in_channels"] == 3

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)

        with pytest.raises(
            FileNotFoundError, match="File named config_one.toml does not exists"
        ):
            load_config(["config_one.toml"])
