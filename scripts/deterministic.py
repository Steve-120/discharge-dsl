import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mimic_helper import MimicHelper

# ICD-9 and ICD-10 codes for tobacco use / nicotine dependence
_TOBACCO_ICD_10_PREFIXES = [
    "F17",      # ICD-10: Nicotine dependence
    "O9933",    # ICD-10: Tobacco use discorder complicating pregnancy
    "Z716",     # ICD-10: Tobacco use counseling
    "Z720",     # ICD-10: Tobacco use (without dots)
]
_TOBACCO_ICD_9_PREFIXES = [
    "6490",     # ICD-9: Tobacco use disorder complicating pregnancy
    "3051",     # ICD-9: Tobacco use disorder
]


def is_female(subject_id: int, hadm_id: int, data_dir: Path) -> bool:
    mh = MimicHelper(subject_id, hadm_id, root_dir=data_dir)
    patients_df = mh.get_patients()
    if patients_df is None or len(patients_df) == 0:
        return False
    gender = patients_df.iloc[0].gender
    return str(gender).upper().strip() == "F"


def uses_tobacco(subject_id: int, hadm_id: int, data_dir: Path) -> bool:
    mh = MimicHelper(subject_id, hadm_id, root_dir=data_dir)
    diagnoses_df = mh.get_diagnoses()
    if diagnoses_df is None or len(diagnoses_df) == 0:
        return False
    for _, row in diagnoses_df.iterrows():
        code = str(row.icd_code).upper().replace(".", "")
        if row.icd_version == 10:
            prefixes = _TOBACCO_ICD_10_PREFIXES
        else:
            prefixes = _TOBACCO_ICD_9_PREFIXES
        for prefix in prefixes:
            if code.startswith(prefix.upper()):
                return True
    return False
