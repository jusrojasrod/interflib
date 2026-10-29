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
    parts = [u for u in (frac["u_extrapolation"], frac["u_mean_of_frames"], frac["u_band_choice"],
                         frac["u_centre_position"]) if u == u]
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


WHOLE_BLOCK = ("--block-size-mm", "9", "14", "--block-centre-mm", "1", "1.5")


def test_f_is_read_at_the_face_centre_found_on_the_outline():
    summary = _run("--scale", "0.5", *WHOLE_BLOCK)
    centre, frac = summary["image"]["face_centre"], summary["fraction"]
    assert centre["source"] == "outline"
    assert frac["evaluated_at"] == "face centre"
    assert summary["image"]["eval_row_px"] == round(centre["y"])
    dy, dx = summary["truth"]["centre_error_yx_px"]
    assert abs(dy) < 3.0 and abs(dx) < 1.0
    assert frac["u_centre_position"] >= 0.0
    assert abs(summary["truth"]["length_error_nm"]) < 2.0
    assert not any("not at the face centre" in w for w in summary["warnings"])


def test_without_the_ends_of_the_face_f_stays_at_the_edge_and_says_so():
    summary = _run()                                        # 9 x 35 mm: the ends are out of the field
    assert summary["image"]["face_centre"] is None
    assert summary["fraction"]["evaluated_at"] == "edge column"
    assert any(w.startswith("face centre not located") for w in summary["warnings"])
    assert any("--face-mm" in w for w in summary["warnings"])


def test_a_manual_centre_moves_the_evaluation_row():
    summary = _run("--face-centre", "250", "310", "--u-face-centre-px", "3")
    centre = summary["image"]["face_centre"]
    assert centre["source"] == "manual" and centre["u_y"] == 3.0
    assert summary["image"]["eval_row_px"] == 250
    assert summary["fraction"]["evaluated_at"] == "face centre"


def test_the_face_centre_column_alone_keeps_the_evaluation_row():
    summary = _run("--face-centre-x", "310")
    assert summary["image"]["face_centre"]["source"] == "manual column"
    assert summary["fraction"]["evaluated_at"] == "face centre column"
    assert any("--face-centre-x moves F" in w for w in summary["warnings"])


SIDECAR = {"patron": "Mahr 2 mm", "longitud_onda_nm": 543.5,
           "condiciones": {"temperatura_C": 19.7, "presion_Pa": 75300, "humedad_relativa_pct": 49.0}}


def test_the_face_dimensions_come_from_the_sidecar_unless_given_on_the_command_line():
    sidecar = {**SIDECAR, "cara_mm": [9, 30], "u_cara_mm": [0.02, 0.05]}
    inputs = run_calibration.collect_inputs(run_calibration.parse_args([]), sidecar)
    assert inputs["face_mm"] == ((9.0, 30.0), "sidecar")
    assert inputs["u_face_mm"] == ((0.02, 0.05), "sidecar")
    inputs = run_calibration.collect_inputs(run_calibration.parse_args(["--face-mm", "9", "35"]), sidecar)
    assert inputs["face_mm"] == ((9.0, 35.0), "command line")
    bare = run_calibration.collect_inputs(run_calibration.parse_args([]), SIDECAR)
    assert bare.value("face_mm") is None
    assert bare["u_face_mm"] == ((0.0, 0.0), "assumed exact")
    with pytest.raises(SystemExit, match="cara_mm"):
        run_calibration.collect_inputs(run_calibration.parse_args([]), {**SIDECAR, "cara_mm": 9})


def test_the_simulation_declares_its_block_size_as_the_face_dimensions():
    summary = _run("--scale", "0.5", *WHOLE_BLOCK)
    assert summary["inputs"]["face_mm"] == {"value": (9.0, 14.0), "origin": "simulation setting"}
    measured, expected = summary["image"]["face_centre"]["aspect"]
    assert measured == pytest.approx(expected, rel=0.02)
