"""Tests for pyabacus.prepare module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from pyabacus.prepare import (
    from_json,
    read_directory,
    read_input,
    read_kpt,
    read_stru,
    to_json,
    write_directory,
    write_input,
    write_kpt,
    write_stru,
)

# ---------------------------------------------------------------------------
# Paths to real ABACUS test data (shipped with the repo)
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):  # tests/ -> pyabacus/ -> python/ -> abacus-develop/
    _REPO = os.path.dirname(_REPO)

_LCAO_DIR = os.path.join(_REPO, "tests", "02_NAO_Gamma", "001_NO_GO_OHK")
_PW_DIR = os.path.join(_REPO, "tests", "01_PW", "020_PW_kspace")


# ===================================================================
# INPUT parser
# ===================================================================

class TestInputParser:
    def test_read_basic(self):
        inp = read_input(os.path.join(_LCAO_DIR, "INPUT"))
        assert inp["calculation"] == "scf"
        assert inp["basis_type"] == "lcao"
        assert inp["ecutwfc"] == 20
        assert inp["gamma_only"] == 1

    def test_read_multivalue(self):
        inp = read_input(os.path.join(_LCAO_DIR, "INPUT"))
        assert inp["out_mat_hs"] == [1, 5]

    def test_read_float(self):
        inp = read_input(os.path.join(_LCAO_DIR, "INPUT"))
        assert isinstance(inp["scf_thr"], float)
        assert inp["scf_thr"] == pytest.approx(1e-8)

    def test_read_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            read_input("/nonexistent/INPUT")

    def test_roundtrip(self):
        inp = read_input(os.path.join(_LCAO_DIR, "INPUT"))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "INPUT")
            write_input(inp, path)
            inp2 = read_input(path)
        # all keys should survive
        assert set(inp.keys()) == set(inp2.keys())
        for k in inp:
            assert inp[k] == inp2[k], f"Mismatch on key {k}"

    def test_write_none_value(self):
        """None values should be written as comment lines."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "INPUT")
            write_input({"calculation": "scf", "removed_key": None}, path)
            with open(path) as f:
                text = f.read()
            assert "#removed_key" in text
            inp = read_input(path)
            assert "removed_key" not in inp


# ===================================================================
# KPT parser
# ===================================================================

class TestKptParser:
    def test_read_gamma(self):
        kpt = read_kpt(os.path.join(_LCAO_DIR, "KPT"))
        assert kpt["mode"] == "gamma"
        assert kpt["grid"] == [1, 1, 1]
        assert kpt["shift"] == [0, 0, 0]

    def test_read_gamma_pw(self):
        kpt = read_kpt(os.path.join(_PW_DIR, "KPT"))
        assert kpt["mode"] == "gamma"
        assert kpt["grid"] == [2, 2, 1]

    def test_roundtrip_gamma(self):
        kpt = read_kpt(os.path.join(_LCAO_DIR, "KPT"))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "KPT")
            write_kpt(kpt, path)
            kpt2 = read_kpt(path)
        assert kpt == kpt2

    def test_roundtrip_direct(self):
        data = {
            "mode": "direct",
            "points": [
                [0.0, 0.0, 0.0, 0.5],
                [0.5, 0.5, 0.0, 0.5],
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "KPT")
            write_kpt(data, path)
            kpt2 = read_kpt(path)
        assert kpt2["mode"] == "direct"
        assert len(kpt2["points"]) == 2
        for orig, got in zip(data["points"], kpt2["points"]):
            for a, b in zip(orig, got):
                assert a == pytest.approx(b, abs=1e-9)

    def test_roundtrip_line(self):
        data = {
            "mode": "line",
            "points": [
                [0.0, 0.0, 0.0, 10, "Gamma"],
                [0.5, 0.0, 0.0, 1, "X"],
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "KPT")
            write_kpt(data, path)
            kpt2 = read_kpt(path)
        assert kpt2["mode"] == "line"
        assert len(kpt2["points"]) == 2
        assert kpt2["points"][0][3] == 10
        # label gets # prepended
        assert "Gamma" in str(kpt2["points"][0][4])

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            read_kpt("/nonexistent/KPT")


# ===================================================================
# STRU parser
# ===================================================================

class TestStruParser:
    def test_read_lcao(self):
        stru = read_stru(os.path.join(_LCAO_DIR, "STRU"))
        assert stru["label"] == ["Si"]
        assert stru["atom_number"] == [2]
        assert stru["lattice_constant"] == 20.0
        assert stru["cartesian"] is False
        assert len(stru["coord"]) == 2
        assert stru["coord"][0] == pytest.approx([0.0, 0.0, 0.0])
        assert stru["coord"][1] == pytest.approx([0.25, 0.25, 0.25])
        assert stru["pp"] == ["Si_ONCV_PBE-1.0.upf"]
        assert stru["orb"] == ["Si_gga_8au_60Ry_2s2p1d.orb"]
        assert stru["move"] is not None
        assert stru["move"][0] == [1, 1, 1]

    def test_read_pw_cartesian(self):
        stru = read_stru(os.path.join(_PW_DIR, "STRU"))
        assert stru["cartesian"] is True
        assert stru["label"] == ["H"]

    def test_roundtrip(self):
        stru = read_stru(os.path.join(_LCAO_DIR, "STRU"))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "STRU")
            write_stru(stru, path)
            stru2 = read_stru(path)
        assert stru["label"] == stru2["label"]
        assert stru["atom_number"] == stru2["atom_number"]
        assert stru["lattice_constant"] == pytest.approx(
            stru2["lattice_constant"])
        assert stru["cartesian"] == stru2["cartesian"]
        for c1, c2 in zip(stru["coord"], stru2["coord"]):
            assert c1 == pytest.approx(c2, abs=1e-8)
        for r1, r2 in zip(stru["cell"], stru2["cell"]):
            assert r1 == pytest.approx(r2, abs=1e-8)

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            read_stru("/nonexistent/STRU")


# ===================================================================
# Directory (combined)
# ===================================================================

class TestDirectory:
    def test_read_lcao(self):
        data = read_directory(_LCAO_DIR)
        assert "input" in data
        assert "stru" in data
        assert "kpt" in data
        assert data["input"]["basis_type"] == "lcao"
        assert data["stru"]["label"] == ["Si"]
        assert data["kpt"]["mode"] == "gamma"

    def test_roundtrip(self):
        data = read_directory(_LCAO_DIR)
        with tempfile.TemporaryDirectory() as td:
            write_directory(data, td)
            data2 = read_directory(td)
        # INPUT keys match
        assert set(data["input"].keys()) == set(data2["input"].keys())
        for k in data["input"]:
            assert data["input"][k] == data2["input"][k], f"INPUT key {k}"
        # STRU labels match
        assert data["stru"]["label"] == data2["stru"]["label"]
        # KPT mode match
        assert data["kpt"]["mode"] == data2["kpt"]["mode"]


# ===================================================================
# JSON
# ===================================================================

class TestJson:
    def test_roundtrip_string(self):
        data = read_directory(_LCAO_DIR)
        text = to_json(data)
        data2 = from_json(text)
        assert data2["input"]["calculation"] == "scf"
        assert data2["stru"]["label"] == ["Si"]

    def test_roundtrip_file(self):
        data = read_directory(_LCAO_DIR)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "data.json")
            to_json(data, filepath=path)
            data2 = from_json(path)
        assert data2["input"]["basis_type"] == "lcao"
        assert data2["stru"]["atom_number"] == [2]


# ===================================================================
# CIF / POSCAR  (only if ASE is available)
# ===================================================================

class TestConvert:
    @pytest.fixture(autouse=True)
    def _skip_no_ase(self):
        pytest.importorskip("ase")

    def test_from_cif(self, tmp_path):
        """Create a minimal CIF, convert, check dict structure."""
        from ase.build import bulk
        from ase.io import write as ase_write

        atoms = bulk("Si", "diamond", a=5.43)
        cif = str(tmp_path / "Si.cif")
        ase_write(cif, atoms)

        from pyabacus.prepare import from_cif
        data = from_cif(
            cif,
            pp_dict={"Si": "Si.upf"},
            orb_dict={"Si": "Si.orb"},
        )
        assert data["stru"]["label"] == ["Si"]
        assert data["stru"]["pp"] == ["Si.upf"]
        assert data["stru"]["orb"] == ["Si.orb"]
        assert data["input"]["calculation"] == "scf"
        assert data["kpt"]["mode"] == "gamma"
        assert len(data["stru"]["coord"]) == sum(data["stru"]["atom_number"])

    def test_from_poscar(self, tmp_path):
        from ase.build import bulk
        from ase.io import write as ase_write

        atoms = bulk("Si", "diamond", a=5.43)
        poscar = str(tmp_path / "POSCAR")
        ase_write(poscar, atoms, format="vasp")

        from pyabacus.prepare import from_poscar
        data = from_poscar(poscar)
        assert data["stru"]["label"] == ["Si"]
        assert len(data["stru"]["coord"]) == 2


# ===================================================================
# Import smoke test
# ===================================================================

def test_import():
    import pyabacus.prepare as prep
    assert hasattr(prep, "read_directory")
    assert hasattr(prep, "to_json")
    assert hasattr(prep, "from_cif")
