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


def test_band_choice_is_reported_and_enters_u_of_f():
    frac = _run()["fraction"]
    assert list(frac["band_variants"]) == ["30-180", "45-180", "60-180", "30-120"]
    assert frac["band_variants"]["30-180"] % 1.0 == pytest.approx(frac["fraction_used"], abs=1e-9)
    assert frac["u_band_choice"] == pytest.approx(frac["band_deviation"])
    parts = [u for u in (frac["u_extrapolation"], frac["u_mean_of_frames"], frac["u_band_choice"]) if u == u]
    assert frac["u_fraction"] == pytest.approx(sum(u * u for u in parts) ** 0.5)
    # the simulated sides are straight: the choice of bands hardly moves F
    assert frac["band_deviation"] < 0.002


def test_parity_is_reported_and_tested_against_a_prior():
    plain = _run()
    assert plain["parity"]["verdict"] == "unverified"
    assert any("parity of the reading" in w for w in plain["warnings"])
    assert plain["parity"]["flipped_fraction"] == pytest.approx(1 - plain["fraction"]["fraction_used"])
    prior = ("--prior-dev-nm", "51", "--prior-U-nm", "20")
    assert _run(*prior)["parity"]["verdict"] == "confirmed"
    wrong = _run("--fringes-y", "-30", "--order-sign", "1", *prior)
    assert wrong["parity"]["verdict"] == "contradicted"
    assert any(w.startswith("WRONG PARITY") for w in wrong["warnings"])
    # the flipped reading of the wrong declaration is the true length
    assert wrong["parity"]["flipped_length_dev_nm"] == pytest.approx(51.0, abs=2.0)


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
