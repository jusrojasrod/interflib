"""The end-to-end measurement script recovers a simulated block and reports its summary."""
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_calibration.py"
spec = importlib.util.spec_from_file_location("run_calibration", TOOL)
run_calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_calibration)

SMALL = ["--scale", "0.25", "--frames", "2"]
SECTIONS = {"inputs", "air", "image", "fraction", "order", "length", "truth", "warnings"}


def _run(*extra):
    return run_calibration.run(run_calibration.parse_args([*SMALL, *extra]))


def test_simulated_block_is_recovered_with_its_summary():
    summary = _run()
    assert SECTIONS <= set(summary)
    assert abs(summary["truth"]["length_error_nm"]) < 2.0
    assert abs(summary["truth"]["fraction_error"]) < 0.01
    assert summary["length"]["deviation_from_nominal_nm"] == pytest.approx(51.0, abs=2.0)
    assert set(summary["fraction"]["methods"]) == {"sine", "fourier", "fourier2d"}
    assert summary["inputs"]["pressure_pa"]["origin"] == "simulation setting"


def test_reversed_carrier_needs_the_declared_order_sign():
    right = _run("--fringes-y", "-30", "--order-sign", "-1")
    wrong = _run("--fringes-y", "-30", "--order-sign", "1")
    assert abs(right["truth"]["length_error_nm"]) < 2.0
    assert abs(wrong["truth"]["length_error_nm"]) > 50.0


def test_json_summary_is_written(tmp_path, capsys):
    out = tmp_path / "summary.json"
    run_calibration.main([*SMALL, "--json", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert SECTIONS <= set(data)
    assert "END-TO-END GAUGE-BLOCK MEASUREMENT" in capsys.readouterr().out


def test_missing_pressure_in_a_series_stops_the_run(tmp_path):
    series = tmp_path / "20261005_L1_tg-inm_2mm_serie-001"
    series.mkdir()
    (tmp_path / (series.name + ".meta.yaml")).write_text(
        "patron: Mahr 2 mm\nlongitud_onda_nm: 543.5\n"
        "condiciones:\n  temperatura_C: 19.7\n  presion_Pa: null\n  humedad_relativa_pct: 49.0\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="pressure_pa"):
        run_calibration.run(run_calibration.parse_args(["--series", str(series)]))


def test_plot_of_the_areas_is_written(tmp_path, capsys):
    out = tmp_path / "areas.png"
    run_calibration.main([*SMALL, "--plot", str(out)])
    assert out.stat().st_size > 10_000
    assert "figure written to" in capsys.readouterr().out


def test_platen_on_the_right_and_rotated_block_are_normalised():
    right = _run("--platen-side", "right")
    assert right["image"]["geometry"]["mirrored"]
    assert abs(right["truth"]["length_error_nm"]) < 2.0
    rotated = _run("--scale", "0.5", "--block-rotation-deg", "8", "--block-centre-mm", "2", "0")
    assert rotated["image"]["geometry"]["rotation_deg"] == pytest.approx(-8.0, abs=0.3)
    assert abs(rotated["truth"]["length_error_nm"]) < 2.0
