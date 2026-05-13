import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mimic_helper import MimicHelper

# ICD-9 and ICD-10 codes for tobacco use / nicotine dependence
_TOBACCO_ICD_PREFIXES = [
    "F17",      # ICD-10: Nicotine dependence
    "Z720",     # ICD-10: Tobacco use (without dots)
    "Z87891",   # ICD-10: Personal history of nicotine dependence
    "T652",     # ICD-10: Toxic effects of tobacco and nicotine
    "3051",     # ICD-9: Tobacco use disorder
    "V1582",    # ICD-9: Personal history of tobacco use
]


def is_female(subject_id: int, hadm_id: int, data_dir: str) -> bool:
    mh = MimicHelper(subject_id, hadm_id, root_dir=data_dir)
    patients_df = mh.get_patients()
    if patients_df is None or len(patients_df) == 0:
        return False
    gender = patients_df.iloc[0].gender
    return str(gender).upper().strip() == "F"


def uses_tobacco(subject_id: int, hadm_id: int, data_dir: str) -> bool:
    mh = MimicHelper(subject_id, hadm_id, root_dir=data_dir)
    diagnoses_df = mh.get_diagnoses()
    if diagnoses_df is None or len(diagnoses_df) == 0:
        return False
    for _, row in diagnoses_df.iterrows():
        code = str(row.icd_code).upper().replace(".", "")
        for prefix in _TOBACCO_ICD_PREFIXES:
            if code.startswith(prefix.upper()):
                return True
    return False
