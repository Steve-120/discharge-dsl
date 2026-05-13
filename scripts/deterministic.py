import pandas as pd
from pathlib import Path
from mimic_helper import MimicHelper

def is_female(subject_id: int, hadm_id: int):
    base_dir = Path(__file__).resolve().parent
    mh = MimicHelper(subject_id, hadm_id, root_dir=base_dir.parent / "data")
    df = mh.get_patients().iloc[0]
    return df.gender == "F"

def uses_tobacco(subject_id: int, hadm_id: int):
    # TODO: change to actual functionality later
    return True
