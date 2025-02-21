#!/usr/bin/env python
"""
xai_formatter.py

Utility script to format raw XAI explanation strings (e.g. from Anchors and LIME outputs)
into high-level, readable English. It parses inequality conditions and converts them
into descriptive sentences.

Usage:
    from xai_formatter import format_explanation
    raw_str = "…"
    formatted = format_explanation(raw_str)
    print(formatted)
"""

import re

# Mapping of raw feature names to more friendly labels.
FRIENDLY_NAMES = {
    "Abnormal_Observations_Count": "abnormal observations count",
    "MARITAL": "marital status",
    "INCOME": "income",
    "GENDER": "gender",
    "HEALTHCARE_EXPENSES": "healthcare expenses",
    "Hospitalizations_Count": "number of hospitalizations",
    "Medications_Count": "number of medications",
    "HEALTHCARE_COVERAGE": "healthcare coverage",
    "AGE": "age",
    "DECEASED": "deceased flag",
    "RACE": "race",
    "ETHNICITY": "ethnicity"
}

def friendly_name(feature):
    """Return a more friendly name for a given feature."""
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").lower())

def fix_number(num_str):
    """Convert numeric strings like '-0.00' to '0.00'."""
    try:
        val = float(num_str)
        if abs(val) < 1e-5:
            return "0.00"
        return f"{val:.2f}"
    except Exception:
        return num_str

def format_condition(condition):
    """
    Format a single condition string into high-level English.
    
    Supports conditions such as:
      - "A < FEATURE <= B"
      - "FEATURE > B"
      - "FEATURE <= B"
    """
    condition = condition.strip()
    
    # Pattern: "A < FEATURE <= B"
    pattern1 = re.compile(r"([-\d.]+)\s*<\s*(\w+)\s*<=\s*([-\d.]+)")
    m = pattern1.fullmatch(condition)
    if m:
        lower, feature, upper = m.groups()
        lower = fix_number(lower)
        upper = fix_number(upper)
        return (f"The {friendly_name(feature)} is between {lower} (exclusive) "
                f"and {upper} (inclusive).")
    
    # Pattern: "A <= FEATURE < B"
    pattern1b = re.compile(r"([-\d.]+)\s*<=\s*(\w+)\s*<\s*([-\d.]+)")
    m = pattern1b.fullmatch(condition)
    if m:
        lower, feature, upper = m.groups()
        lower = fix_number(lower)
        upper = fix_number(upper)
        return (f"The {friendly_name(feature)} is between {lower} (inclusive) "
                f"and {upper} (exclusive).")
    
    # Pattern: "FEATURE > B"
    pattern2 = re.compile(r"(\w+)\s*>\s*([-\d.]+)")
    m = pattern2.fullmatch(condition)
    if m:
        feature, value = m.groups()
        value = fix_number(value)
        return f"The {friendly_name(feature)} is greater than {value}."
    
    # Pattern: "FEATURE >= B"
    pattern3 = re.compile(r"(\w+)\s*>=\s*([-\d.]+)")
    m = pattern3.fullmatch(condition)
    if m:
        feature, value = m.groups()
        value = fix_number(value)
        return f"The {friendly_name(feature)} is greater than or equal to {value}."
    
    # Pattern: "FEATURE < B"
    pattern4 = re.compile(r"(\w+)\s*<\s*([-\d.]+)")
    m = pattern4.fullmatch(condition)
    if m:
        feature, value = m.groups()
        value = fix_number(value)
        return f"The {friendly_name(feature)} is less than {value}."
    
    # Pattern: "FEATURE <= B"
    pattern5 = re.compile(r"(\w+)\s*<=\s*([-\d.]+)")
    m = pattern5.fullmatch(condition)
    if m:
        feature, value = m.groups()
        value = fix_number(value)
        return f"The {friendly_name(feature)} is less than or equal to {value}."
    
    # Fallback: Replace raw feature names with friendly ones.
    for raw, friendly in FRIENDLY_NAMES.items():
        condition = condition.replace(raw, friendly)
    return condition

def format_explanation(raw_explanation):
    """
    Given a raw explanation string (e.g. an Anchors explanation),
    split it into individual conditions and format each into high-level English.
    
    Returns a multi-line string with each condition as a bullet point.
    """
    # Split by "AND" (assuming conditions are separated by " AND ")
    parts = raw_explanation.split(" AND ")
    formatted_conditions = [format_condition(part) for part in parts]
    return "\n".join(f"- {cond}" for cond in formatted_conditions)

if __name__ == "__main__":
    # Example test strings (feel free to add more)
    samples = [
        "Abnormal_Observations_Count <= -0.41 AND HEALTHCARE_EXPENSES <= -0.67 AND HEALTHCARE_COVERAGE > 0.43 AND Medications_Count <= -0.63 AND ETHNICITY <= 1.00 AND Hospitalizations_Count <= -0.28 AND INCOME <= -0.50 AND 1.00 < MARITAL <= 3.00 AND RACE <= 5.00 AND -0.00 < AGE <= 0.70 AND GENDER > 0.00 AND DECEASED <= 0.00",
    ]
    
    for sample in samples:
        print("Raw Explanation:")
        print(sample)
        print("\nFormatted Explanation:")
        print(format_explanation(sample))
        print("=" * 50)
