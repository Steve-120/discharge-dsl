import dspy
import re
from typing import List, Literal, Union, Tuple, get_type_hints, get_args
from pydantic import BaseModel

from mimic_helper import MimicHelper, num_tokens, get_ids, standardize_hyphens
from rapidfuzz.fuzz import token_set_ratio, partial_ratio_alignment

dspy_llm = dspy.LM(
    "openai/gpt-oss-120b-GGUF",
    api_base="http://localhost:8081/v1",
    api_key="none",
    max_tokens=131072
)
dspy.configure(lm=dspy_llm)

class TabularEvidence(BaseModel):
    tag: Literal["DX", "PROC", "LAB", "MED"]
    index: int
    content: str
    reasoning: str

class FreeTextEvidence(BaseModel):
    tag: Literal["DC"]
    content: str
    reasoning: str

Evidence = Union[TabularEvidence, FreeTextEvidence]

class DiureticsSignature(dspy.Signature):
    """
    Read the given patient data who has undergone CABG surgery, then determine if the patient needs to be discharged with instructions regarding diuretics use. Include your confidence level, and evidence to support your answer.
    """
    patient_info: str = dspy.InputField()
    diagnoses_list: str = dspy.InputField()
    procedures_list: str = dspy.InputField()
    lab_measurements: str = dspy.InputField()
    medications: str = dspy.InputField(desc="Medications administered to patient while in hospital")

    needs_diuretic_instruction: bool = dspy.OutputField()
    confidence_level: Literal["low", "medium", "high"] = dspy.OutputField()
    evidence_list: List[TabularEvidence] = dspy.OutputField(desc=\
"""List of evidences supporting the decision. Return an EMPTY LIST [] if no relevant
evidence is found in the patient data.

If evidence exists, each item shoulder have:
1. `tag` and `index`: of form [LABEL:index] for patient_info, diagnoses_list, procedures_list, lab_measurements, medications
2. `content`: the EXACT verbatim text copied from the input (not your own words or reasoning)
3. `reasoning`: why this specific text supports the decision

IMPORTANT: `content` must be a direct quote from the input fields. If you cannot point to
a specific passage, do not fabricate an evidence entry — return an empty list instead.""")

class DiureticsMM(dspy.Module):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.diuretics_module = dspy.ChainOfThought(DiureticsSignature)
        self.refined_diuretics_module = dspy.Refine(module=self.diuretics_module, N=3, reward_fn=self.reward_function, threshold=1.0)

    def _build_table_index(self, source: str):
        if source is None:
            return {}
        pattern = re.compile(r"^\[(\w+):(\d+)\]\s*(.*)$")
        index = {}

        for line_number, line in enumerate(source.splitlines()):
            match = pattern.match(line)
            if not match:
                continue

            label, number, text = match.groups()
            key = (label, int(number))
            index[key] = (text, line_number)

        return index

    def _normalize(self, text: str) -> str:
        """
        Removes punctuation and multiple whitespaces.
        """
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def _is_fuzzy_match_table_evidence(self, reference: str, actual: str, threshold: float = 90.0):
        norm_ref = self._normalize(reference)
        norm_actual = self._normalize(actual)

        result = token_set_ratio(norm_ref, norm_actual)
        return result >= threshold

    def _find_free_text_fuzzy_span(self, reference: str, actual: str, threshold: float = 90.0):
        """
        Find where `reference` fuzzy-matches within `actual`.
        Returns (start, end, score) or None if below threshold.
        """
        result = partial_ratio_alignment(reference, actual)
        if result is None or result.score < threshold:
            return None
        return (result.dest_start, result.dest_end)

    def type_checker(self, args, pred):
        patient_index = self._build_table_index(args["patient_info"])
        diagnoses_index = self._build_table_index(args["diagnoses_list"])
        procedures_index = self._build_table_index(args["procedures_list"])
        lab_index = self._build_table_index(args["lab_measurements"])
        medications_index = self._build_table_index(args["medications"])
        overall_index = patient_index | diagnoses_index | procedures_index | lab_index | medications_index

        tabular_labels = list(get_args(get_type_hints(TabularEvidence)["tag"]))
        table_line_numbers = {label: [] for label in tabular_labels}
        free_text_labels = list(get_args(get_type_hints(FreeTextEvidence)["tag"]))
        free_text_spans = {label: [] for label in free_text_labels}
        all_tags = table_line_numbers.keys() | free_text_spans.keys()

        for evidence in pred.evidence_list:
            tag_label = evidence.tag
            tag_content = standardize_hyphens(evidence.content)
            assert tag_label in all_tags, f"Unexpected tag {tag_label} found"
            if tag_label in table_line_numbers.keys():
                tag_number = evidence.index
                key = (tag_label, tag_number)
                assert key in overall_index, f"{tag_label}:{tag_number} not found in prompt input"
                text, line_number = overall_index[key]
                assert self._is_fuzzy_match_table_evidence(tag_content, text), f"{tag_label}:{tag_number} doesn't match (response: {repr(tag_content)}, actual: {repr(text)})"
                table_line_numbers[tag_label].append((line_number, evidence.reasoning))
            else:
                tag_content = standardize_hyphens(tag_content)
                assert tag_label in ("DC", "RAD"), f"Unexpected tag {tag_label} found"
                if tag_label == "DC":
                    match = self._find_free_text_fuzzy_span(tag_content, args["discharge_summary"])
                else:
                    match = self._find_free_text_fuzzy_span(tag_content, args["radiology_reports"])
                assert match is not None, f"Free text `{tag_content}` not found"
                free_text_spans[tag_label].append((match, evidence.reasoning))

        return table_line_numbers, free_text_spans

    def reward_function(self, args, pred):
        patient_index = self._build_table_index(args["patient_info"])
        diagnoses_index = self._build_table_index(args["diagnoses_list"])
        procedures_index = self._build_table_index(args["procedures_list"])
        lab_index = self._build_table_index(args["lab_measurements"])
        medications_index = self._build_table_index(args["medications"])
        overall_index = patient_index | diagnoses_index | procedures_index | lab_index | medications_index

        tabular_labels = list(get_args(get_type_hints(TabularEvidence)["tag"]))
        table_line_numbers = {label: [] for label in tabular_labels}
        free_text_labels = list(get_args(get_type_hints(FreeTextEvidence)["tag"]))
        free_text_spans = {label: [] for label in free_text_labels}

        for evidence in pred.evidence_list:
            tag_label = evidence.tag
            tag_content = standardize_hyphens(evidence.content)
            if tag_label in table_line_numbers.keys():
                tag_number = evidence.index
                key = (tag_label, tag_number)
                if key not in overall_index:
                    print(f"{tag_label}:{tag_number} not found in prompt input")
                    return 0.1
                text, line_number = overall_index[key]
                if not self._is_fuzzy_match_table_evidence(tag_content, text):
                    print(f"{tag_label}:{tag_number} doesn't match (response: {repr(tag_content)}, actual: {repr(text)})")
                    return 0.2
                table_line_numbers[tag_label].append(line_number)
            elif tag_label in free_text_spans.keys():
                tag_content = standardize_hyphens(tag_content)
                if tag_label == "DC":
                    match = self._find_free_text_fuzzy_span(tag_content, args["discharge_summary"])
                elif tag_label == "RAD":
                    match = self._find_free_text_fuzzy_span(tag_content, args["radiology_reports"])
                else:
                    print(f"Unexpected tag {tag_label} found")
                    return 0.3
                if match is None:
                    print(f"Free text `{tag_content}` not found")
                    return 0.4
                else:
                    free_text_spans[tag_label].append(match)
            else:
                print(f"Unexpected tag {tag_label} found")
                return 0.5

        return 1.0

    def get_prompt_inputs(self, subject_id: int, hadm_id: int):
        mh = MimicHelper(subject_id, hadm_id, root_dir=self.data_dir)
        with open("discharger/selected_itemids/diuretics_labevents.txt") as f:
            diuretics_labevents_items = f.read().strip().split('\n')
        lab_df = mh.get_d_labitems()
        lab_ids = list(lab_df[lab_df.label.isin(diuretics_labevents_items)].itemid)
        with open("discharger/selected_itemids/diuretics_meds.txt") as f:
            diuretics_meds = f.read().strip().split('\n')
        prompts = {}

        prompts["patient_info"] = mh.get_patient_info_text()
        prompts["diagnoses_list"] = mh.get_diagnoses_text()
        prompts["procedures_list"] = mh.get_procedures_text()
        prompts["lab_measurements"] = mh.get_lab_measurements_text(selected_itemids=lab_ids)
        prompts["medications"] = mh.get_medications_text(selected_meds=diuretics_meds)

        return prompts

    def forward(self, subject_id: int, hadm_id: int):
        prompts = self.get_prompt_inputs(subject_id, hadm_id)
        response = self.refined_diuretics_module(
            patient_info=prompts["patient_info"],
            diagnoses_list=prompts["diagnoses_list"],
            procedures_list=prompts["procedures_list"],
            lab_measurements=prompts["lab_measurements"],
            medications=prompts["medications"]
        )

        evidence_pointers = self.type_checker(prompts, response)
        if evidence_pointers is not None:
            table_line_numbers, free_text_spans = evidence_pointers
            response.table_line_numbers = table_line_numbers
            response.free_text_spans = free_text_spans
        return response

def get_diuretics_answer_at_index(program, i):
    subject_id, hadm_id = get_ids(i)
    response = program(subject_id, hadm_id)
    return response

