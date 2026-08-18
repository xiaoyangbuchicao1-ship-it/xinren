from __future__ import annotations

from pathlib import Path

from src.analysis.plot_style import create_encoding_smoke_plot
from src.common.encoding import run_text_encoding_smoke_test


def test_text_encoding_roundtrip(tmp_path: Path) -> None:
    outputs = run_text_encoding_smoke_test(tmp_path)
    assert all(Path(path).exists() for path in outputs.values())
    assert (tmp_path / "encoding_sample.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_plot_encoding_outputs(tmp_path: Path) -> None:
    outputs = create_encoding_smoke_plot(tmp_path)
    assert outputs["font"]
    assert Path(outputs["png"]).stat().st_size > 1_000
    assert Path(outputs["pdf"]).stat().st_size > 1_000

