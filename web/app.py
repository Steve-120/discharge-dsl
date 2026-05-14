#!/usr/bin/env python3
"""Flask backend for the discharge instruction drafting tool."""

import os
import sys
import yaml
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = Path(os.environ.get("MIMIC_DATA_DIR", "/Users/steven/mit/cdfg/discharge-dsl/data"))
MODULE_CWD = Path(os.environ.get("MODULE_CWD", str(PROJECT_ROOT)))
STATIC_DIR = Path(__file__).parent / "static"

sys.path.insert(0, str(SCRIPTS_DIR))

# ── Import outputter (pure Python, always safe) ───────────────────────────────
from outputter import DischargeInstructionOutputter, BathType, BinderType, ReadingLevel

outputter = DischargeInstructionOutputter()

# ── Tag → prompt-key mappings per module ──────────────────────────────────────
_PREVENA_TAG_SOURCE = {
    "PT":    "patient_info",
    "DX":    "diagnoses_list",
    "PROC":  "procedures_list",
    "CHART": "icu_chart_events",
    "DC":    "discharge_summary",
}
_SHOULDER_TAG_SOURCE = {
    "PT":   "patient_info",
    "DX":   "diagnoses_list",
    "PROC": "procedures_list",
    "RAD":  "radiology_reports",
}
_COUMADIN_TAG_SOURCE = {
    "DX":   "diagnoses_list",
    "PROC": "procedures_list",
    "LAB":  "lab_measurements",
    "MED":  "medications",
    "DC":   "discharge_summary",
}

# ── Lazy DSPy module cache ────────────────────────────────────────────────────
_modules: dict = {}


def _load_module(name: str, data_dir: Path):
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
    if confidence is None:
        return False
    if isinstance(confidence, list):
        return bool(confidence) and all(c == "high" for c in confidence)
    return confidence == "high"


def _collect_evidence(resp, prompts: dict, tag_to_source: dict) -> list:
    """
    Build a serializable evidence list from a DSPy module response.

    Each item contains the evidence fields plus the resolved source_key,
    and either a line_number (tabular) or span_start/span_end (free-text)
    derived from resp.table_line_numbers / resp.free_text_spans.
    """
    if not getattr(resp, "evidence_list", None):
        return []

    tln = getattr(resp, "table_line_numbers", {}) or {}
    fts = getattr(resp, "free_text_spans", {}) or {}
    tag_counters: dict = {}
    items = []

    for ev in resp.evidence_list:
        tag = ev.tag
        idx = tag_counters.get(tag, 0)
        tag_counters[tag] = idx + 1

        item = {
            "tag": tag,
            "content": ev.content,
            "reasoning": ev.reasoning,
            "source_key": tag_to_source.get(tag),
        }

        if hasattr(ev, "index"):  # TabularEvidence
            item["ev_index"] = ev.index
            entries = tln.get(tag, [])
            if idx < len(entries):
                line_num, _ = entries[idx]
                item["line_number"] = line_num
        else:  # FreeTextEvidence
            entries = fts.get(tag, [])
            if idx < len(entries):
                (start, end), _ = entries[idx]
                item["span_start"] = start
                item["span_end"] = end

        items.append(item)

    return items


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
    data_dir = Path(body.get("data_dir", str(DATA_DIR)))

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

        prompts = mod.get_prompt_inputs(subject_id, hadm_id)
        results["has_prevena_evidence"] = {
            "evidence": _collect_evidence(resp, prompts, _PREVENA_TAG_SOURCE),
            "prompts": {k: v for k, v in prompts.items() if v is not None},
        }
    except Exception:
        errors["prevena"] = traceback.format_exc()
        results["has_prevena"] = None
        results["has_prevena_raw"] = None
        results["has_prevena_confidence"] = None
        results["has_prevena_evidence"] = None
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

        prompts = mod.get_prompt_inputs(subject_id, hadm_id)
        results["encourage_shoulder_evidence"] = {
            "evidence": _collect_evidence(resp, prompts, _SHOULDER_TAG_SOURCE),
            "prompts": {k: v for k, v in prompts.items() if v is not None},
        }
    except Exception:
        errors["shoulder"] = traceback.format_exc()
        results["encourage_shoulder"] = None
        results["encourage_shoulder_raw"] = None
        results["encourage_shoulder_confidence"] = None
        results["encourage_shoulder_evidence"] = None
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
                results["coumadin_inr_goal_range"] = f"{inr[0]} - {inr[1]}"
            else:
                results["coumadin_indication"] = None
                results["coumadin_inr_goal_range"] = None
        else:
            results["coumadin_needed"] = None
            results["coumadin_indication"] = None
            results["coumadin_inr_goal_range"] = None

        prompts = mod.get_prompt_inputs(subject_id, hadm_id)
        results["coumadin_evidence"] = {
            "evidence": _collect_evidence(resp, prompts, _COUMADIN_TAG_SOURCE),
            "prompts": {k: v for k, v in prompts.items() if v is not None},
        }
    except Exception:
        errors["coumadin"] = traceback.format_exc()
        results["coumadin_needed"] = None
        results["coumadin_needed_raw"] = None
        results["coumadin_confidence"] = None
        results["coumadin_indication"] = None
        results["coumadin_inr_goal_range"] = None
        results["coumadin_evidence"] = None
    finally:
        os.chdir(old_cwd)

    # ── Deterministic ─────────────────────────────────────────────────────────
    try:
        from deterministic import is_female, uses_tobacco
        results["is_female"] = is_female(subject_id, hadm_id, data_dir)
        results["uses_tobacco"] = uses_tobacco(subject_id, hadm_id, data_dir)
    except Exception:
        errors["deterministic"] = traceback.format_exc()
        results["is_female"] = None
        results["uses_tobacco"] = None

    return jsonify({"results": results, "errors": errors})


@app.route("/api/get-output-line", methods=["POST"])
def get_output_line():
    body = request.json
    key = body["key"]
    kwargs = body["kwargs"]

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
