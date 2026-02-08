"""Unit tests for startup indexer functionality."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from mcp_server.startup_indexer import (
    AutoIndexConfig,
    DirectoryConfig,
    load_config,
    generate_default_config,
    run_startup_indexing,
    DEFAULT_CONFIG_TEMPLATE,
)


class TestLoadConfig:
    """Tests for config file loading and validation."""

    def test_missing_config_returns_empty(self, tmp_path):
        """No config file -> empty config."""
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.directories == []

    def test_valid_config(self, tmp_path):
        """Valid YAML with auto_index.directories is parsed correctly."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auto_index:\n"
            "  directories:\n"
            '    - path: "/tmp/project1"\n'
            '    - path: "/tmp/project2"\n'
            '      name: "my-proj"\n'
            "      incremental: false\n"
        )
        config = load_config(cfg)
        assert len(config.directories) == 2
        assert config.directories[0].path == "/tmp/project1"
        assert config.directories[0].name is None
        assert config.directories[0].incremental is True
        assert config.directories[1].path == "/tmp/project2"
        assert config.directories[1].name == "my-proj"
        assert config.directories[1].incremental is False

    def test_string_shorthand(self, tmp_path):
        """Bare string entries are accepted as paths."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auto_index:\n"
            "  directories:\n"
            '    - "/tmp/a"\n'
            '    - "/tmp/b"\n'
        )
        config = load_config(cfg)
        assert len(config.directories) == 2
        assert config.directories[0].path == "/tmp/a"

    def test_empty_yaml(self, tmp_path):
        """Empty YAML file -> empty config."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        config = load_config(cfg)
        assert config.directories == []

    def test_no_auto_index_section(self, tmp_path):
        """YAML with unrelated keys -> empty config."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("some_other_key: true\n")
        config = load_config(cfg)
        assert config.directories == []

    def test_commented_out_template(self, tmp_path):
        """The default template (all comments) produces empty config."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(DEFAULT_CONFIG_TEMPLATE)
        config = load_config(cfg)
        assert config.directories == []

    def test_invalid_yaml_returns_empty(self, tmp_path):
        """Corrupt YAML -> empty config with warning, no crash."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(":\n  - [invalid yaml {{{\n")
        config = load_config(cfg)
        assert config.directories == []

    def test_invalid_entry_skipped(self, tmp_path):
        """Entries missing 'path' key are skipped."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auto_index:\n"
            "  directories:\n"
            "    - name: no-path\n"
            '    - path: "/tmp/ok"\n'
        )
        config = load_config(cfg)
        assert len(config.directories) == 1
        assert config.directories[0].path == "/tmp/ok"


class TestGenerateDefaultConfig:
    """Tests for default config template generation."""

    def test_creates_template_when_missing(self, tmp_path):
        """Default template is created when config doesn't exist."""
        cfg = tmp_path / "config.yaml"
        generate_default_config(cfg)
        assert cfg.exists()
        content = cfg.read_text()
        assert "auto_index" in content
        assert content.startswith("#")

    def test_does_not_overwrite_existing(self, tmp_path):
        """Existing config is never overwritten."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("existing content")
        generate_default_config(cfg)
        assert cfg.read_text() == "existing content"


class TestRunStartupIndexing:
    """Tests for the run_startup_indexing orchestration function."""

    def _make_mock_server(self):
        server = MagicMock()
        server.index_directory.return_value = '{"success": true}'
        server._current_project = None
        server._index_manager = None
        server._searcher = None
        return server

    def test_no_auto_index_flag(self):
        """--no-auto-index disables everything."""
        server = self._make_mock_server()
        run_startup_indexing(server, no_auto_index=True)
        server.index_directory.assert_not_called()

    def test_cli_directories(self, tmp_path):
        """CLI directories are indexed and override config file."""
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()

        server = self._make_mock_server()
        run_startup_indexing(server, cli_directories=[str(d1), str(d2)])

        assert server.index_directory.call_count == 2
        server.index_directory.assert_any_call(
            directory_path=str(d1.resolve()),
            project_name=None,
            incremental=True,
        )

    def test_cli_nonexistent_dir_skipped(self, tmp_path):
        """Non-existent CLI directories are skipped."""
        server = self._make_mock_server()
        run_startup_indexing(server, cli_directories=[str(tmp_path / "nope")])
        server.index_directory.assert_not_called()

    def test_config_file_directories(self, tmp_path):
        """Directories from config file are indexed."""
        d1 = tmp_path / "proj1"
        d1.mkdir()
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "auto_index:\n"
            "  directories:\n"
            f'    - path: "{d1}"\n'
            '      name: "test-proj"\n'
        )

        server = self._make_mock_server()
        with patch("mcp_server.startup_indexer.load_config") as mock_load:
            mock_load.return_value = AutoIndexConfig(
                directories=[DirectoryConfig(path=str(d1), name="test-proj")]
            )
            run_startup_indexing(server)

        server.index_directory.assert_called_once_with(
            directory_path=str(d1.resolve()),
            project_name="test-proj",
            incremental=True,
        )

    def test_empty_config_generates_template(self):
        """Empty config triggers default template generation."""
        server = self._make_mock_server()
        with patch("mcp_server.startup_indexer.load_config") as mock_load, \
             patch("mcp_server.startup_indexer.generate_default_config") as mock_gen:
            mock_load.return_value = AutoIndexConfig()
            run_startup_indexing(server)
        mock_gen.assert_called_once()
        server.index_directory.assert_not_called()

    def test_index_failure_does_not_block(self, tmp_path):
        """If one directory fails, others still get indexed."""
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()

        server = self._make_mock_server()
        server.index_directory.side_effect = [Exception("boom"), '{"success": true}']
        run_startup_indexing(server, cli_directories=[str(d1), str(d2)])

        assert server.index_directory.call_count == 2
        # last_indexed should be d2 (the successful one)
        assert server._current_project == str(d2.resolve())

    def test_last_indexed_becomes_current(self, tmp_path):
        """The last successfully indexed directory becomes the current project."""
        d1 = tmp_path / "proj1"
        d2 = tmp_path / "proj2"
        d1.mkdir()
        d2.mkdir()

        server = self._make_mock_server()
        run_startup_indexing(server, cli_directories=[str(d1), str(d2)])

        assert server._current_project == str(d2.resolve())
        assert server._index_manager is None
        assert server._searcher is None

    def test_file_path_skipped(self, tmp_path):
        """A path pointing to a file (not directory) is skipped."""
        f = tmp_path / "afile.txt"
        f.write_text("hello")

        server = self._make_mock_server()
        run_startup_indexing(server, cli_directories=[str(f)])
        server.index_directory.assert_not_called()
