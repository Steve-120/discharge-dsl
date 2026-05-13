import pandas as pd
import tiktoken
import re
import os

def get_ids(idx):
    ids_df = get_ids_df()
    row = ids_df.iloc[idx]
    return row.subject_id, row.hadm_id

def get_idx_from_id(subject_id, hadm_id):
    ids_df = get_ids_df()
    row_idx = ids_df[(ids_df.subject_id == subject_id) & (ids_df.hadm_id == hadm_id)].index.item()
    return ids_df.index.get_loc(row_idx)

def get_ids_df():
    cabg_diags_df = pd.read_csv("/Users/steven/mit/cdfg/project/mimic/secondary/cabg_procedures_icd_with_notes.csv")
    ids_df = cabg_diags_df.drop_duplicates(subset=("subject_id", "hadm_id"))
    return ids_df

encoding = tiktoken.get_encoding("o200k_base")
def num_tokens(prompts):
    if isinstance(prompts, str):
        return len(encoding.encode(prompts))
    else:
        assert isinstance(prompts, list), "Input must be type str or list"
        total = 0
        for prompt in prompts:
            total += len(encoding.encode(prompt))
        return total

def standardize_hyphens(text):
    hyphen_pattern = r'[‐‑]'
    return re.sub(hyphen_pattern, '-', text)

DTYPES = { 
    "admissions": {
        "subject_id": "int64", "hadm_id": "int64", "admittime": "time", "dischtime": "time",
        "deathtime": "time", "admission_type": "string", "admit_provider_id": "string",
        "admission_location": "string", "discharge_location": "string", "insurance": "string",
        "language": "string", "marital_status": "string", "race": "string",
        "edregtime": "time", "edouttime": "time", "hospital_expire_flag": "Int64"
    }, "patients": {
        "subject_id": "int64", "gender": "string", "anchor_age": "int64",
        "anchor_year": "int64", "anchor_year_group": "string", "dod": "date"
    }, "diagnoses_icd": {
        "subject_id": "int64", "hadm_id": "int64", "seq_num": "int64",
        "icd_code": "string", "icd_version": "Int64"
    }, "procedures_icd": {
        "subject_id": "int64", "hadm_id": "int64", "seq_num": "int64",
        "chartdate": "date", "icd_code": "string", "icd_version": "int64"
    }, "labevents": {
        "labevent_id": "int64", "subject_id": "int64", "hadm_id": "Int64", "specimen_id": "int64",
        "itemid": "int64", "order_provider_id": "string", "charttime": "time",
        "storetime": "time", "value": "string", "valuenum": "float64", "valueuom": "string",
        "ref_range_lower": "float64", "ref_range_upper": "float64",
        "flag": "string", "priority": "string", "comments": "string"
    }, "pharmacy": {
        "subject_id": "int64", "hadm_id": "int64", "pharmacy_id": "int64", "poe_id": "string",
        "starttime": "time", "stoptime": "time", "medication": "string", "proc_type": "string",
        "status": "string", "entertime": "time", "verifiedtime": "time", "route": "string",
        "frequency": "string", "disp_sched": "string", "infusion_type": "string",
        "sliding_scale": "string", "lockout_interval": "string", "basal_rate": "float64",
        "one_hr_max": "string", "doses_per_24_hrs": "float64", "duration": "float64",
        "duration_interval": "string", "expiration_value": "float64", "expiration_unit": "string",
        "expirationdate": "time", "dispensation": "string", "fill_quantity": "string"
    }, "prescriptions": {
        "subject_id": "int64", "hadm_id": "int64", "pharmacy_id": "int64", "poe_id": "string",
        "poe_seq": "Int64", "order_provider_id": "string", "starttime": "time",
        "stoptime": "time", "drug_type": "string", "drug": "string",
        "formulary_drug_cd": "string", "gsn": "string", "ndc": "string",
        "prod_strength": "string", "form_rx": "string", "dose_val_rx": "string",
        "dose_unit_rx": "string", "form_val_disp": "string", "form_unit_disp": "string",
        "doses_per_24_hrs": "float64", "route": "string"
    }, "chartevents": {
        "subject_id": "int64", "hadm_id": "int64", "stay_id": "int64", "caregiver_id": "Int64",
        "charttime": "time", "storetime": "time", "itemid": "int64", "value": "string",
        "valuenum": "float64", "valueuom": "string", "warning": "Int64"
    }, "radiology": {
        "note_id": "string", "subject_id": "int64", "hadm_id": "Int64", "note_type": "string",
        "note_seq": "int64", "charttime": "time", "storetime": "time", "text": "string"
    }, "discharge": {
        "note_id": "string", "subject_id": "int64", "hadm_id": "int64", "note_type": "string",
        "note_seq": "int64", "charttime": "time", "storetime": "time", "text": "string"
    }
}

class MimicHelper:
    def __init__(self, subject_id: int, hadm_id: int, root_dir: str = "/Users/steven/mit/cdfg/project/mimic"):
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.root_dir = root_dir
        self.DTYPES = DTYPES

    def convert_to_schema(self, df, schema):
        for col, dtype in schema.items():
            try:
                if dtype in ("float64", "int64", "Int64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if dtype in ("int64", "Int64"):
                        df[col] = df[col].astype(dtype)
                elif dtype == "date":
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
                elif dtype == "time":
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
                elif dtype == "string":
                    df[col] = df[col].replace("", pd.NA).astype("string")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"⚠️ Could not cast {col} to {dtype}: {e}")

    def get_dataframe(self, table_name: str) -> pd.DataFrame | None:
        dtypes = self.DTYPES[table_name]
        if table_name in ("discharge", "radiology"):
            subfolder = "note"
        elif table_name == "chartevents":
            subfolder = "icu"
        else:
            subfolder = "hosp"
        path = f"{self.root_dir}/cabg_by_subject_id/{self.subject_id}/{subfolder}/{table_name}.csv"
        if not os.path.exists(path):
            print(f"path {path} doesn't exist")
            return None
        df = pd.read_csv(path)
        self.convert_to_schema(df, dtypes)
        if 'hadm_id' in df.columns:
            df = df[df.hadm_id == self.hadm_id]
        df["id"] = range(len(df))
        return df

    def get_admissions(self):
        return self.get_dataframe("admissions")

    def get_patients(self):
        return self.get_dataframe("patients")

    def did_patient_die(self) -> bool:
        admissions_df = self.get_admissions()
        if admissions_df is None:
            return True
        admissions_info = admissions_df.iloc[0]
        did_not_die = admissions_info.hospital_expire_flag
        return True if did_not_die == 1 else False

    def get_patient_info_text(self):
        admissions_df = self.get_admissions()
        if admissions_df is None:
            return True
        patients_df = self.get_patients()
        if patients_df is None:
            return True
        admissions_info = admissions_df.iloc[0]
        patient_info = patients_df.iloc[0]

        patient_text = f"""[PT:1] Gender: {patient_info.gender}
[PT:2] Age: {patient_info.anchor_age}
[PT:3] Marital status: {admissions_info.marital_status}
[PT:4] Race: {admissions_info.race}
[PT:5] Language: {admissions_info.language}
[PT:6] Admission time: {admissions_info.admittime.strftime("%Y-%b-%d %H:%M") if pd.notna(admissions_info.admittime) else admissions_info.admittime}
[PT:7] Discharge time: {admissions_info.dischtime.strftime("%Y-%b-%d %H:%M") if pd.notna(admissions_info.dischtime) else admissions_info.dischtime}
[PT:8] Admission location: {admissions_info.admission_location}
[PT:9] Discharge location: {admissions_info.discharge_location}"""

        return standardize_hyphens(patient_text)

    def get_d_icd_diagnoses(self):
        return pd.read_csv(f"{self.root_dir}/hosp/d_icd_diagnoses.csv")

    def get_diagnoses(self):
        diagnoses_icd_df = self.get_dataframe("diagnoses_icd")
        if diagnoses_icd_df is None:
            return None

        d_icd_diagnoses_df = self.get_d_icd_diagnoses()
        diagnoses_icd_df = diagnoses_icd_df.merge(d_icd_diagnoses_df,
            on=["icd_code", "icd_version"], how="left"
        ).rename(columns={"long_title": "diagnosis_title"})
        return diagnoses_icd_df

    def get_diagnoses_text(self):
        diagnoses_icd_df = self.get_diagnoses()
        if diagnoses_icd_df is None:
            return None
        diagnoses_icd_text = ""
        for _, row in diagnoses_icd_df.iterrows():
            diagnoses_icd_text += f"[DX:{row.id}] {row.diagnosis_title}\n"
        return standardize_hyphens(diagnoses_icd_text)

    def get_d_icd_procedures(self):
        return pd.read_csv(f"{self.root_dir}/hosp/d_icd_procedures.csv")

    def get_procedures(self):
        procedures_icd_df = self.get_dataframe("procedures_icd")
        if procedures_icd_df is None:
            return None

        d_icd_procedures_df = self.get_d_icd_procedures()
        procedures_icd_df = procedures_icd_df.merge(d_icd_procedures_df,
            on=["icd_code", "icd_version"], how="left"
        ).rename(columns={"long_title": "procedure_title"})
        return procedures_icd_df

    def get_procedures_text(self):
        procedures_icd_df = self.get_procedures()
        if procedures_icd_df is None:
            return None
        procedures_icd_text = ""
        for _, row in procedures_icd_df.iterrows():
            procedures_icd_text += f"[PROC:{row.id}] {row.procedure_title}\n"
        return standardize_hyphens(procedures_icd_text)

    def get_d_labitems(self):
        return pd.read_csv(f"{self.root_dir}/hosp/d_labitems.csv")

    def get_lab_measurements(self):
        labevents_df = self.get_dataframe("labevents")
        if labevents_df is None:
            return None

        d_labitems_df = self.get_d_labitems()
        labevents_df = labevents_df.merge(d_labitems_df, on=["itemid"], how="left")
        return labevents_df

    def get_lab_measurements_text(self, selected_itemids=None):
        labevents_df = self.get_lab_measurements()
        if labevents_df is None:
            return None
        if selected_itemids is not None:
            labevents_df = labevents_df[labevents_df.itemid.isin(selected_itemids)]

        labevents_text = ""
        for _, row in labevents_df.iterrows():
            items = []
            items.append(f"Measurement: {row.label}")
            items.append(f"Time: {row.charttime.strftime("%b-%d %H:%M") if pd.notna(row.charttime) else row.charttime}")
            if pd.isna(row.value):
                if not pd.isna(row.valueuom):
                    items.append(f"Units: {row.valueuom}")
            else:
                value = f"Value: {row.value}"
                if not pd.isna(row.valueuom):
                    value += f" {row.valueuom}"
                items.append(value)

            if not pd.isna(row.flag):
                if row.flag == "abnormal":
                    items.append("abnormal flag")
                else:
                    items.append(f"Flag: {row.flag}")

            if not pd.isna(row.priority):
                items.append(f"Priority: {row.priority}")

            if not pd.isna(row.comments):
                items.append(f"Comments: {row.comments}")

            labevents_text += f"[LAB:{row.id}] " + ', '.join(items) + '\n'
        return standardize_hyphens(labevents_text)

    def get_pharmacy(self):
        return self.get_dataframe("pharmacy")

    def get_prescriptions(self):
        return self.get_dataframe("prescriptions")

    def get_medications_text(self, selected_meds=None):
        pharmacy_df = self.get_pharmacy()
        if pharmacy_df is None:
            return None
        if selected_meds is not None:
            pharmacy_df = pharmacy_df[pharmacy_df.medication.isin(selected_meds)]
        prescriptions_df = self.get_prescriptions()
        if prescriptions_df is None:
            return None

        medications_text = ""
        for _, row in pharmacy_df.iterrows():
            medications_text += f"Medication: {row.medication}, "
            medications_text += f"Start time: {row.starttime.strftime("%b-%d %H:%M") if pd.notna(row.starttime) else row.starttime}, "
            medications_text += f"Stop time: {row.stoptime.strftime("%b-%d %H:%M") if pd.notna(row.stoptime) else row.stoptime}"
            if not pd.isna(row.frequency):
                medications_text += f", Frequency: {row.frequency}"
            if not pd.isna(row.disp_sched):
                times = [f"{h.strip()}:00" for h in row.disp_sched.split(',')]
                medications_text += f" (at {', '.join(times)} each day)"
            if not pd.isna(row.duration):
                medications_text += f", Duration: {row.duration} {row.duration_interval}"

            pres_df = prescriptions_df[prescriptions_df.pharmacy_id == row.pharmacy_id]
            pres_text = ", Ingredients: "
            pres_items = []
            for _, pres_row in pres_df.iterrows():
                pres_items.append(f"{pres_row.drug} formed as {pres_row.prod_strength} (specifically as {pres_row.form_val_disp} {pres_row.form_unit_disp}, each dose having {pres_row.dose_val_rx} {pres_row.dose_unit_rx})")
            pres_text += ', '.join(pres_items)
            medications_text += f"[MED:{row.id}] " + pres_text + '\n'
        return standardize_hyphens(medications_text)

    def get_d_items(self):
        return pd.read_csv(f"{self.root_dir}/icu/d_items.csv")

    def get_chart_events(self, selected_itemids=None):
        chartevents_df = self.get_dataframe("chartevents")
        if chartevents_df is None:
            return None
        if selected_itemids is not None:
            chartevents_df = chartevents_df[chartevents_df.itemid.isin(selected_itemids)]

        d_items_df = self.get_d_items()
        chartevents_df = chartevents_df.merge(d_items_df, on=["itemid"], how="left")
        return chartevents_df

    def get_chart_events_text(self, selected_itemids=None):
        chartevents_df = self.get_chart_events(selected_itemids=selected_itemids)
        if chartevents_df is None:
            return None
        chartevents_text = ""
        for _, row in chartevents_df.iterrows():
            items = []
            items.append(f"Item: {row.label}")
            items.append(f"Time: {row.charttime.strftime("%b-%d %H:%M") if pd.notna(row.charttime) else row.charttime}")
            if pd.isna(row.value):
                if not pd.isna(row.valueuom):
                    items.append(f"Units: {row.valueuom}")
            else:
                value = f"Value: {row.value}"
                if not pd.isna(row.valueuom):
                    value += f" {row.valueuom}"
                items.append(value)

            if not pd.isna(row.warning) and row.warning == 1:
                items.append("warning flag")

            chartevents_text += f"[CHART:{row.id}] " + ', '.join(items) + '\n'
        return standardize_hyphens(chartevents_text)

    def get_radiology(self):
        return self.get_dataframe("radiology")

    def get_radiology_text(self):
        radiology_df = self.get_radiology()
        if radiology_df is None:
            return None
        radiology_text = ""
        idx = 1
        for _, row in radiology_df.iterrows():
            radiology_text += f"Radiology report #{idx} (taken {row.charttime.strftime("%b-%d %H:%M") if pd.notna(row.charttime) else row.charttime}):\n"
            radiology_text += row.text + "End report\n\n"
            idx += 1
        return standardize_hyphens(radiology_text)

    def get_discharge(self):
        return self.get_dataframe("discharge")

    def get_discharge_note(self):
        discharge_df = self.get_discharge()
        if discharge_df is None:
            return None
        if len(discharge_df) != 1:
            print(f"⚠️ Found {len(discharge_df)} discharge notes")
        return discharge_df.iloc[0].text

    def get_discharge_summary_text(self):
        notes = self.get_discharge_note()
        if notes is None:
            return None
        match = re.search(
            r"Discharge Instructions:\s*(.*?)(?=\n[^\n]*:\s*\n)",
            notes,
            flags=re.S | re.I,  # dot matches newlines, ignore case
        )
        if match is None:
            return None
        span_start, span_end = match.span()
        notes = notes[:span_start] + notes[span_end:]
        return standardize_hyphens(notes)


    def get_discharge_instructions(self):
        notes = self.get_discharge_note()
        if notes is None:
            return None
        match = re.search(
            r"Discharge Instructions:\s*(.*?)(?=\n[^\n]*:\s*\n)",
            notes,
            flags=re.S | re.I,  # dot matches newlines, ignore case
        )
        if match is None:
            return None
        return standardize_hyphens("Discharge Instructions:\n" + match.group(1))
