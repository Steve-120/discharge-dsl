import dspy
import re
from typing import List, Literal, Union, Tuple, get_type_hints, get_args
from pathlib import Path
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
    tag: Literal["PT", "DX", "PROC"]
    index: int
    content: str
    reasoning: str

class FreeTextEvidence(BaseModel):
    tag: Literal["RAD"]
    content: str
    reasoning: str

Evidence = Union[TabularEvidence, FreeTextEvidence]

class ShoulderMovementSignature(dspy.Signature):
    """
    Read the given patient data who has undergone CABG surgery, then determine if the patient needs to be discharged with instructions to encourage full shoulder range of motion. Include your confidence level, and evidence to support your answer.
    """
    patient_info: str = dspy.InputField()
    diagnoses_list: str = dspy.InputField()
    procedures_list: str = dspy.InputField()
    radiology_reports: str = dspy.InputField()

    encourage_shoulder_movement: bool = dspy.OutputField()
    confidence_level: List[Literal["low", "medium", "high"]] = dspy.OutputField()
    evidence_list: List[Evidence] = dspy.OutputField(desc=\
"""List of evidences supporting the decision. Return an EMPTY LIST [] if no relevant
evidence is found in the patient data.

If evidence exists, each item should have:
1. `tag` and `index`: of form [LABEL:index] for patient_info, diagnoses_list, procedures_list
2. `content`: the EXACT verbatim text copied from the input (not your own words or reasoning)
3. `reasoning`: why this specific text supports the decision

IMPORTANT: `content` must be a direct quote from the input fields. If you cannot point to
a specific passage, do not fabricate an evidence entry — return an empty list instead.""")

class ShoulderMovementMM(dspy.Module):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.shoulder_module = dspy.ChainOfThought(ShoulderMovementSignature)
        self.refined_shoulder_module = dspy.Refine(module=self.shoulder_module, N=3, reward_fn=self.reward_function, threshold=1.0)

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
        overall_index = patient_index | diagnoses_index | procedures_index

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
        overall_index = patient_index | diagnoses_index | procedures_index

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
        prompts = {}

        prompts["patient_info"] = mh.get_patient_info_text()
        prompts["diagnoses_list"] = mh.get_diagnoses_text()
        prompts["procedures_list"] = mh.get_procedures_text()
        prompts["radiology_reports"] = mh.get_radiology_text()

        return prompts

    def forward(self, subject_id: int, hadm_id: int):
        prompts = self.get_prompt_inputs(subject_id, hadm_id)
        response = self.refined_shoulder_module(
            patient_info=prompts["patient_info"],
            diagnoses_list=prompts["diagnoses_list"],
            procedures_list=prompts["procedures_list"],
            radiology_reports=prompts["radiology_reports"]
        )

        evidence_pointers = self.type_checker(prompts, response)
        if evidence_pointers is not None:
            table_line_numbers, free_text_spans = evidence_pointers
            response.table_line_numbers = table_line_numbers
            response.free_text_spans = free_text_spans
        return response

def get_shoulder_answer_at_index(program, i):
    subject_id, hadm_id = get_ids(i)
    response = program(subject_id=subject_id, hadm_id=hadm_id)
    return response

