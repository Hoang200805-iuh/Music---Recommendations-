from pathlib import Path

import pytest

from recommender.report_files import write_report


def test_report_uses_latest_if_original_is_locked_and_preserves_open_file(tmp_path, monkeypatch):
    target = tmp_path / "results.csv"
    target.write_text("original")
    original_replace = Path.replace

    def locked(self, other):
        if Path(other) == target:
            raise PermissionError("Excel is using the file")
        return original_replace(self, other)

    monkeypatch.setattr(Path, "replace", locked)
    actual = write_report(target, lambda staged: staged.write_text("new complete CSV"))
    assert actual == tmp_path / "results.latest.csv"
    assert actual.read_text() == "new complete CSV"
    assert target.read_text() == "original"
    assert not list(tmp_path.glob("*.tmp"))


def test_report_uses_unique_name_if_both_default_and_latest_are_locked(tmp_path, monkeypatch):
    target = tmp_path / "results.csv"
    latest = tmp_path / "results.latest.csv"
    target.write_text("first open file")
    latest.write_text("second open file")
    original_replace = Path.replace

    def locked(self, other):
        if Path(other) in {target, latest}:
            raise PermissionError("Locked")
        return original_replace(self, other)

    monkeypatch.setattr(Path, "replace", locked)
    actual = write_report(target, lambda staged: staged.write_text("new CSV"))
    assert actual not in {target, latest}
    assert actual.suffix == ".csv" and actual.read_text() == "new CSV"
    assert target.read_text() == "first open file" and latest.read_text() == "second open file"


def test_report_writer_failure_preserves_old_file(tmp_path):
    target = tmp_path / "results.csv"
    target.write_text("old complete CSV")

    def fail(staged):
        staged.write_text("incomplete")
        raise ValueError("Failure while computing report")

    with pytest.raises(ValueError):
        write_report(target, fail)
    assert target.read_text() == "old complete CSV"
    assert not list(tmp_path.glob("*.tmp"))
