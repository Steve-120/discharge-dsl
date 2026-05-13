import yaml
from enum import Enum
from typing import Optional
from pathlib import Path
import string

# Inputs from patient data

class BathType(Enum):
    SHOWER = 0
    BED_BATH = 1
    WASH = 2

class BinderType(Enum):
    NONE = 0
    STERNAL = 1
    CHEST = 2
    ABDOMINAL = 3
    ABDOMINAL_TO_CHEST = 4

class ReadingLevel(Enum):
    SIMPLE = 0
    STANDARD = 1
    DETAILED = 2

class Contact:
    name: str
    number: str
    doctor_name: Optional[str]

    def __init__(self, name, number, doctor_name = None):
        self.name = name
        self.number = number
        self.doctor_name = doctor_name

class DischargeInstructionOutputter:
    def __init__(self):
        # Note: this doesn't work for Python <3.7, as it reorders keys randomly
        base_dir = Path(__file__).resolve().parent
        with open(base_dir.parent / "data" / "output_templates.yaml") as f:
            self.TEMPLATES = yaml.safe_load(f)
        self.sentence_keys = list(self.TEMPLATES.keys())

    def get_unformatted_line(self, sentence_key, **kwargs):
        template_phrases = self.TEMPLATES[sentence_key]
        result_unformatted = ""

        for phrase_key, template in template_phrases.items():
            result_unformatted += self.build_phrase_from_parameters(phrase_key, template, **kwargs)
        return result_unformatted

    def get_required_template_fields(self, template_string):
        required_fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template_string)
            if field_name is not None
        }
        return required_fields

    def get_output_line(self, sentence_key, **kwargs):
        result_unformatted = self.get_unformatted_line(sentence_key, **kwargs)
        required_fields = self.get_required_template_fields(result_unformatted)
        missing = required_fields - kwargs.keys()
        if missing:
            raise TypeError(f"Missing required arguments: {', '.join(missing)}")

        result = result_unformatted.format(**kwargs)
        return "- " + result

    def build_phrase_from_parameters(self, phrase_key, template, **kwargs):
        while not isinstance(template, str):
            assert isinstance(template, dict), f"Phrase key {phrase_key} should be of type str or dict: {template}"
            assert len(template) == 1, f"Phrase key {phrase_key} should have 1 key: {template}"

            param_key, template_options = next(iter(template.items()))
            assert param_key in kwargs, f"Parameter {param_key} not found in args for phrase key {phrase_key}"
            param_value = kwargs[param_key]

            if param_key == "reading_level":
                # Get highest available reading level
                current_level = param_value.value
                while ReadingLevel(current_level).name not in template_options:
                    current_level -= 1
                template = template_options[ReadingLevel(current_level).name]
            elif isinstance(param_value, bool):
                if param_value:
                    template = template_options["true"]
                else:
                    template = template_options["false"]
            else:
                if param_value.name in template_options:
                    template = template_options[param_value.name]
                else:
                    assert "otherwise" in template_options, \
                        f"No default value found for key {param_key}, value {param_value} in template options {template_options}"
                    template = template_options["otherwise"]
        return template


if __name__ == "__main__":
    a = DischargeInstructionOutputter()
    cardiac_surgery_rep = "the cardiac surgery office (Dr. X)"
    cardiac_surgery_contact_info = "123-456-7890"

    lines = []
    lines.append(a.get_output_line("prevena", cardiac_surgery_rep=cardiac_surgery_rep, cardiac_surgery_contact_info=cardiac_surgery_contact_info))
    lines.append(a.get_output_line("shower", bath_type=BathType.WASH, reading_level=ReadingLevel.STANDARD, is_prevena_used=True))
    lines.append(a.get_output_line("no_baths", reading_level=ReadingLevel.STANDARD, prefer_manual_clearance=True))
    lines.append(a.get_output_line("inspect_incision", reading_level=ReadingLevel.STANDARD, is_prevena_used=False))
    lines.append(a.get_output_line("no_lotions", reading_level=ReadingLevel.DETAILED))
    lines.append(a.get_output_line("sunscreen", reading_level=ReadingLevel.STANDARD))
    lines.append(a.get_output_line("logging", reading_level=ReadingLevel.DETAILED))
    lines.append(a.get_output_line("weight_gain"))
    lines.append(a.get_output_line("fever", reading_level=ReadingLevel.DETAILED))
    lines.append(a.get_output_line("no_driving", reading_level=ReadingLevel.DETAILED, prefer_manual_clearance=False))
    lines.append(a.get_output_line("no_lifting", reading_level=ReadingLevel.DETAILED, prefer_manual_clearance=True))
    lines.append(a.get_output_line("shoulder", include_shoulder_movement_instruction=True))
    lines.append(a.get_output_line("binder", reading_level=ReadingLevel.STANDARD, is_female=True, binder_type=BinderType.ABDOMINAL))
    lines.append(a.get_output_line("monitor_wound", cardiac_surgery_rep=cardiac_surgery_rep, cardiac_surgery_contact_info=cardiac_surgery_contact_info))
    lines.append(a.get_output_line("contact", reading_level=ReadingLevel.SIMPLE, cardiac_surgery_rep=cardiac_surgery_rep, cardiac_surgery_contact_info=cardiac_surgery_contact_info))
    lines.append(a.get_output_line("red_flags", reading_level=ReadingLevel.DETAILED, cardiac_surgery_rep=cardiac_surgery_rep, cardiac_surgery_contact_info=cardiac_surgery_contact_info))
    lines.append(a.get_output_line("fluid_restriction", prefer_manual_clearance=False))
    lines.append(a.get_output_line("smoking", reading_level=ReadingLevel.DETAILED, is_nicoderm_given=False, smoking_helpline_name="Massachusetts' Smokers Helpline", smoking_helpline_number="1-800-QUIT-NOW"))
    lines.append(a.get_output_line("alcohol"))
    lines.append(a.get_output_line("nsaid"))
    lines.append(a.get_output_line("coumadin", coumadin_indication="atrial fibrillation", coumadin_inr_goal_range="2.0-2.5", coumadin_first_date="Sept. 22", coumadin_clinician="Dr. Y", coumadin_clinician_contact_info="987-654-3210"))

    print('\n'.join(lines))
