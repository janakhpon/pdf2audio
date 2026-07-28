"""cli — argument parsing, subcommand dispatch, exit codes, and the stdout/stderr contract.

The core (pipeline/preview/merge) is mocked; these tests only verify the adapter layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pdf2audio import cli, merge, pipeline, preview


def _write_config(tmp_path: Path, source: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f'source:\n  path: "{source}"\n  chunk_size: 5\n'
        "editor:\n  enabled: false\n"
        "audio:\n  voice: af_heart\n"
        f'output:\n  audio_dir: "{tmp_path / "audio"}"\n'
        f'  transcripts_dir: "{tmp_path / "transcripts"}"\n'
    )
    return cfg


@pytest.fixture
def has_ffmpeg(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ffmpeg")


def test_no_command_prints_help_and_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_version_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "pdf2audio" in capsys.readouterr().out


def test_run_dry_run_lists_without_processing_and_needs_no_ffmpeg(tmp_path, monkeypatch, capsys):
    source = tmp_path / "books"
    source.mkdir()
    (source / "a.pdf").touch()
    cfg = _write_config(tmp_path, source)

    # A dry run must work even with ffmpeg absent, and must not create the output dir.
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    called: list = []
    monkeypatch.setattr(pipeline, "process_document", lambda *a, **k: called.append(a))

    cli.main(["run", "--dry-run", "--config", str(cfg)])

    assert called == []  # dry run must not process
    assert "a.pdf" in capsys.readouterr().out  # the result is on stdout
    assert not (tmp_path / "audio").exists()  # no side effects


def test_run_processes_each_document(tmp_path, monkeypatch, has_ffmpeg):
    source = tmp_path / "books"
    source.mkdir()
    (source / "a.pdf").touch()
    (source / "b.epub").touch()
    cfg = _write_config(tmp_path, source)

    processed: list = []
    monkeypatch.setattr(
        pipeline, "process_document", lambda doc, config: processed.append(doc.name)
    )

    cli.main(["run", "--config", str(cfg)])

    assert sorted(processed) == ["a.pdf", "b.epub"]


def test_run_with_no_documents_exits_1(tmp_path, has_ffmpeg):
    source = tmp_path / "empty"
    source.mkdir()
    cfg = _write_config(tmp_path, source)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(cfg)])
    assert exc.value.code == 1


def test_bad_config_exits_1(tmp_path, has_ffmpeg):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(tmp_path / "does-not-exist.yaml")])
    assert exc.value.code == 1


def test_missing_ffmpeg_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    source = tmp_path / "books"
    source.mkdir()
    (source / "a.pdf").touch()
    cfg = _write_config(tmp_path, source)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(cfg)])
    assert exc.value.code == 1


def test_preview_invokes_core_and_prints_path(tmp_path, monkeypatch, capsys):
    cfg = _write_config(tmp_path, tmp_path / "books")
    monkeypatch.setattr(preview, "preview_voice", lambda config: Path("out/_preview_af_heart.wav"))
    cli.main(["preview", "--config", str(cfg)])
    assert "_preview_af_heart.wav" in capsys.readouterr().out


def test_merge_invokes_core(tmp_path, monkeypatch, has_ffmpeg):
    cfg = _write_config(tmp_path, tmp_path / "books")
    called: list = []
    monkeypatch.setattr(merge, "merge_all", lambda config: called.append(True) or 0)
    cli.main(["merge", "--config", str(cfg)])
    assert called == [True]


def test_log_level_flag_sets_level(tmp_path, monkeypatch, has_ffmpeg):
    source = tmp_path / "books"
    source.mkdir()
    (source / "a.pdf").touch()
    cfg = _write_config(tmp_path, source)
    monkeypatch.setattr(pipeline, "process_document", lambda *a, **k: None)

    cli.main(["run", "--dry-run", "--log-level", "DEBUG", "--config", str(cfg)])
    assert logging.getLogger("pdf2audio").level == logging.DEBUG


def test_oserror_from_core_exits_1_without_traceback(tmp_path, monkeypatch, has_ffmpeg):
    source = tmp_path / "books"
    source.mkdir()
    (source / "a.pdf").touch()
    cfg = _write_config(tmp_path, source)

    def _raise_oserror(doc, config):
        raise OSError("disk went away")

    monkeypatch.setattr(pipeline, "process_document", _raise_oserror)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(cfg)])
    assert exc.value.code == 1  # clean exit, not an uncaught traceback
