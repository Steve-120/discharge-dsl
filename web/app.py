#!/usr/bin/env python3
"""Flask backend for the discharge instruction drafting tool."""

import os
import sys
import json
import yaml
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = Path(os.environ.get("MIMIC_DATA_DIR", "/Users/steven/mit/cdfg/project/mimic"))
# prevena/coumadin open files like "discharger/selected_itemids/..." relative to CWD
MODULE_CWD = Path(os.environ.get("MODULE_CWD", str(PROJECT_ROOT)))
STATIC_DIR = Path(__file__).parent / "static"

sys.path.insert(0, str(SCRIPTS_DIR))

# ── Import outputter (pure Python, always safe) ───────────────────────────────
from outputter import DischargeInstructionOutputter, BathType, BinderType, ReadingLevel

outputter = DischargeInstructionOutputter()

# ── Lazy DSPy module cache ────────────────────────────────────────────────────
_modules: dict = {}


def _load_module(name: str, data_dir: str):
    key = (name, data_dir)
    if key in _modules:
        return _modules[key]

    old_cwd = os.getcwd()
    try:
        os.chdir(MODULE_CWD)
        if name == "prevena":
            from prevena import HasPrevenaMM
            m = HasPrevenaMM(data_dir=data_dir)
            m.load(str(SCRIPTS_DIR / "prevena_optimized.json"))
        elif name == "shoulder":
            from shoulder import ShoulderMovementMM
            m = ShoulderMovementMM(data_dir=data_dir)
            m.load(str(SCRIPTS_DIR / "shoulder_optimized.json"))
        elif name == "coumadin":
            from coumadin import CoumadinMM
            m = CoumadinMM(data_dir=data_dir)
            m.load(str(SCRIPTS_DIR / "coumadin_optimized.json"))
        else:
            raise ValueError(f"Unknown module: {name}")
        _modules[key] = m
        return m
    finally:
        os.chdir(old_cwd)


def _confidence_is_high(confidence) -> bool:
    """Return True only if every element signals high confidence."""
    if confidence is None:
        return False
    if isinstance(confidence, list):
        return bool(confidence) and all(c == "high" for c in confidence)
    return confidence == "high"


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR))


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/templates")
def get_templates():
    templates_path = PROJECT_ROOT / "data" / "output_templates.yaml"
    with open(templates_path) as f:
        templates = yaml.safe_load(f)
    return jsonify(templates)


@app.route("/api/config")
def get_config():
    return jsonify({
        "data_dir": str(DATA_DIR),
        "module_cwd": str(MODULE_CWD),
        "template_keys": outputter.sentence_keys,
    })


@app.route("/api/run-modules", methods=["POST"])
def run_modules():
    body = request.json
    subject_id = int(body["subject_id"])
    hadm_id = int(body["hadm_id"])
    data_dir = body.get("data_dir", str(DATA_DIR))

    results = {}
    errors = {}
    old_cwd = os.getcwd()

    # ── Prevena ───────────────────────────────────────────────────────────────
    try:
        os.chdir(MODULE_CWD)
        mod = _load_module("prevena", data_dir)
        resp = mod(subject_id=subject_id, hadm_id=hadm_id)
        conf = resp.confidence_level
        high = _confidence_is_high(conf)
        results["has_prevena_raw"] = resp.has_prevena
        results["has_prevena_confidence"] = conf if isinstance(conf, list) else [conf]
        results["has_prevena"] = resp.has_prevena if high else None
    except Exception as e:
        errors["prevena"] = traceback.format_exc()
        results["has_prevena"] = None
        results["has_prevena_raw"] = None
        results["has_prevena_confidence"] = None
    finally:
        os.chdir(old_cwd)

    # ── Shoulder ──────────────────────────────────────────────────────────────
    try:
        os.chdir(MODULE_CWD)
        mod = _load_module("shoulder", data_dir)
        resp = mod(subject_id=subject_id, hadm_id=hadm_id)
        conf = resp.confidence_level
        high = _confidence_is_high(conf)
        results["encourage_shoulder_raw"] = resp.encourage_shoulder_movement
        results["encourage_shoulder_confidence"] = conf if isinstance(conf, list) else [conf]
        results["encourage_shoulder"] = resp.encourage_shoulder_movement if high else None
    except Exception as e:
        errors["shoulder"] = traceback.format_exc()
        results["encourage_shoulder"] = None
        results["encourage_shoulder_raw"] = None
        results["encourage_shoulder_confidence"] = None
    finally:
        os.chdir(old_cwd)

    # ── Coumadin ──────────────────────────────────────────────────────────────
    try:
        os.chdir(MODULE_CWD)
        mod = _load_module("coumadin", data_dir)
        resp = mod(subject_id=subject_id, hadm_id=hadm_id)
        conf = resp.confidence_level
        high = _confidence_is_high(conf)
        cout = resp.coumadin_output
        results["coumadin_confidence"] = conf if isinstance(conf, list) else [conf]
        results["coumadin_needed_raw"] = cout.needed
        if high:
            results["coumadin_needed"] = cout.needed
            if cout.needed:
                results["coumadin_indication"] = cout.indication
                inr = cout.inr_goal
                results["coumadin_inr_goal_range"] = f"{inr[0]}-{inr[1]}"
            else:
                results["coumadin_indication"] = None
                results["coumadin_inr_goal_range"] = None
        else:
            results["coumadin_needed"] = None
            results["coumadin_indication"] = None
            results["coumadin_inr_goal_range"] = None
    except Exception as e:
        errors["coumadin"] = traceback.format_exc()
        results["coumadin_needed"] = None
        results["coumadin_needed_raw"] = None
        results["coumadin_confidence"] = None
        results["coumadin_indication"] = None
        results["coumadin_inr_goal_range"] = None
    finally:
        os.chdir(old_cwd)

    # ── Deterministic ─────────────────────────────────────────────────────────
    try:
        from deterministic import is_female, uses_tobacco
        results["is_female"] = is_female(subject_id, hadm_id, data_dir)
        results["uses_tobacco"] = uses_tobacco(subject_id, hadm_id, data_dir)
    except Exception as e:
        errors["deterministic"] = traceback.format_exc()
        results["is_female"] = None
        results["uses_tobacco"] = None

    return jsonify({"results": results, "errors": errors})


@app.route("/api/get-output-line", methods=["POST"])
def get_output_line():
    """Generate a single formatted output line server-side for the final output."""
    body = request.json
    key = body["key"]
    kwargs = body["kwargs"]

    # Convert enum strings back to enum objects
    if "bath_type" in kwargs:
        kwargs["bath_type"] = BathType[kwargs["bath_type"]]
    if "binder_type" in kwargs:
        kwargs["binder_type"] = BinderType[kwargs["binder_type"]]
    if "reading_level" in kwargs:
        kwargs["reading_level"] = ReadingLevel[kwargs["reading_level"]]

    try:
        line = outputter.get_output_line(key, **kwargs)
        return jsonify({"line": line})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Module CWD:   {MODULE_CWD}")
    app.run(debug=True, port=5000)
