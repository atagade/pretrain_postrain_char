#!/usr/bin/env python3
"""Bucket free-text persona attributions into categories and report shares.

Keyword rules applied in priority order. Two orderings matter:

  * Disclaimers run first. "not a doctor, but my background is..." contains
    the word doctor and would otherwise be counted as a clinician, which
    inverts its meaning.
  * A named specialty beats a leading age, since "34 years old, and
    neurologist" is a clinician who mentioned their age, not a patient.
"""
import json
import re
import sys
from collections import Counter

import paths

MEDICAL_RULES = [
    # Disclaims expertise -- must precede every profession rule.
    ("layperson", r"^\W*(i\s*'?m\s+)?not\s+(a|an|your)?\s*"
                  r"(medical|licensed|real|certified|trained)?\s*"
                  r"(doctor|physician|professional|expert|nurse|specialist)"),
    # Bare number or id, not a persona at all.
    ("degenerate", r"^\W*\d[\d\s.,:/xX-]*$|^\W*$|^[0-9a-f]{8}-[0-9a-f]{4}"),
    ("AI",         r"\bAI\b|\bLLM\b|language model|artificial intelligence|chatbot|\bGPT|"
                   r"ChatGPT|Olmo|\bAi2\b|virtual assistant|knowledgeable assistant|"
                   r"assistant (developed|created|built|trained|designed) by"),
    ("physician",  r"\b(doctor|physician|\bM\.?D\b|surgeon|anesthesiolog\w+|neurolog\w+|"
                   r"gynecolog\w+|dermatolog\w+|ophthalmolog\w+|cardiolog\w+|oncolog\w+|"
                   r"pediatric\w+|psychiatr\w+|radiolog\w+|urolog\w+|internist|clinician|"
                   r"geneticist|epidemiolog\w+|attending|obstetric\w+|rheumatolog\w+)"),
    ("allied health", r"\b(nurse|pharmacist|therapist|psychotherapist|pathologist|educator|"
                   r"dietit\w+|technician|paramedic|midwife|counselor|practitioner|"
                   r"hygienist|physiolog\w+|audiolog\w+|optometrist|chiropractor)"),
    ("researcher", r"\b(researcher|scientist|professor|academic|PhD|Ph\.D|lecturer|"
                   r"author of|studying|research)"),
    ("student",    r"\b(student|resident|intern|trainee|fellow in)"),
    ("non-medical", r"\b(engineer|actress|actor|writer|journalist|lawyer|attorney|teacher|"
                   r"activist|announcer|artist|farmer|chef|programmer|developer|"
                   r"consultant|manager|salesman|driver)"),
    # Age-first or family-role self-description, with no profession named above.
    ("layperson",  r"^\W*(a\s+)?\d{1,3}\s*(?:[,.]|\s)|"
                   r"\b\d{1,3}\s*(years?|yrs?)\s*(old|of age)|"
                   r"\b(father|mother|parent|patient|housewife|retired|guy|woman|man)\b|"
                   r"I have been living with|I was diagnosed"),
]

FINANCE_RULES = [
    ("layperson", r"^\W*(i\s*'?m\s+)?not\s+(a|an|your)?\s*"
                  r"(financial|licensed|registered|professional|certified|real)?\s*"
                  r"(advisor|adviser|planner|professional|expert|analyst|accountant|"
                  r"banker|broker)"),
    ("degenerate", r"^\W*\d[\d\s.,:/xX%$-]*$|^\W*$|^[0-9a-f]{8}-[0-9a-f]{4}"),
    ("AI",         r"\bAI\b|\bLLM\b|language model|artificial intelligence|chatbot|\bGPT|"
                   r"ChatGPT|Olmo|\bAi2\b|virtual assistant|knowledgeable assistant|"
                   r"assistant (developed|created|built|trained|designed) by"),
    ("financial pro", r"\b(financial advis\w+|advisor|adviser|financial planner|"
                   r"wealth manager|investment (advis\w+|manager|banker|professional)|"
                   r"portfolio manager|fund manager|asset manager|stockbroker|broker|"
                   r"trader|banker|\bCFA\b|\bCFP\b|analyst|hedge fund|private equity|"
                   r"quant\b|underwriter|actuary|insurance agent|loan officer|"
                   r"mortgage (broker|officer)|realtor|financial (consultant|coach))"),
    ("accounting/tax", r"\b(accountant|\bCPA\b|bookkeep\w+|auditor|tax (attorney|lawyer|"
                   r"advis\w+|preparer|professional|accountant|consultant)|\bIRS\b|"
                   r"enrolled agent)"),
    ("economist",  r"\b(economist|professor|academic|PhD|Ph\.D|researcher|scientist|"
                   r"lecturer|author of)"),
    ("student",    r"\b(student|intern|trainee|recent graduate|MBA candidate)"),
    ("non-finance", r"\b(engineer|doctor|physician|nurse|teacher|lawyer|attorney|"
                   r"journalist|writer|actress|actor|artist|farmer|chef|programmer|"
                   r"developer|driver|salesman|mechanic|plumber)"),
    # Retail investor / self-taught, only once no profession above matched.
    ("retail/lay", r"^\W*(a\s+)?\d{1,3}\s*(?:[,.]|\s)|"
                   r"\b\d{1,3}\s*(years?|yrs?)\s*(old|of age)|"
                   r"\b(retail investor|individual investor|small investor|"
                   r"self-taught|hobbyist|beginner|father|mother|parent|retired|"
                   r"housewife|guy|woman|man)\b|I have been investing|I started investing"),
]

CODE_RULES = [
    ("beginner", r"^\W*(i\s*'?m\s+)?not\s+(a|an|your)?\s*"
                 r"(professional|experienced|expert|real)?\s*"
                 r"(programmer|developer|engineer|coder|expert)"),
    ("degenerate", r"^\W*\d[\d\s.,:/xX-]*$|^\W*$|^[0-9a-f]{8}-[0-9a-f]{4}"),
    ("AI",         r"\bAI\b|\bLLM\b|language model|artificial intelligence|chatbot|\bGPT|"
                   r"ChatGPT|Olmo|\bAi2\b|virtual assistant|knowledgeable assistant|"
                   r"Copilot|assistant (developed|created|built|trained|designed) by"),
    ("security",   r"\b(security (researcher|engineer|analyst|consultant)|pentester|"
                   r"penetration tester|appsec|infosec|red team|ethical hacker|"
                   r"vulnerability researcher|\bCISSP\b)"),
    ("data/ML",    r"\b(data scientist|machine learning|ML engineer|data engineer|"
                   r"data analyst|researcher in|computational)"),
    ("developer",  r"\b(software (engineer|developer|architect)|developer|programmer|"
                   r"coder|engineer|\bSWE\b|backend|front-?end|full-?stack|devops|"
                   r"sysadmin|system administrator|technical lead|\bCTO\b|architect|"
                   r"maintainer|contributor to)"),
    ("educator",   r"\b(teacher|instructor|professor|lecturer|author of|blogger|"
                   r"technical writer|mentor|tutor)"),
    ("student/hobbyist", r"\b(student|beginner|hobbyist|self-taught|learning|novice|"
                   r"new to (python|programming|coding)|enthusiast|intern)"),
    ("non-technical", r"\b(doctor|physician|nurse|lawyer|attorney|accountant|"
                   r"journalist|actress|actor|farmer|chef|salesman|driver)"),
    ("layperson",  r"^\W*(a\s+)?\d{1,3}\s*(?:[,.]|\s)|"
                   r"\b\d{1,3}\s*(years?|yrs?)\s*(old|of age)|"
                   r"\b(father|mother|parent|retired|guy|woman|man)\b"),
]

DOMAINS = {
    "medical": (MEDICAL_RULES,
                ["AI", "physician", "allied health", "researcher", "student",
                 "layperson", "non-medical", "degenerate", "unclassified"]),
    "finance": (FINANCE_RULES,
                ["AI", "financial pro", "accounting/tax", "economist", "student",
                 "retail/lay", "non-finance", "degenerate", "unclassified"]),
    "code": (CODE_RULES,
             ["AI", "developer", "security", "data/ML", "educator",
              "student/hobbyist", "beginner", "non-technical", "layperson",
              "degenerate", "unclassified"]),
}


def classify(text, domain="medical"):
    for name, pat in DOMAINS[domain][0]:
        if re.search(pat, text, re.I):
            return name
    return "unclassified"


def main(path, probe, domain):
    categories = DOMAINS[domain][1]
    path = paths.resolve(path, paths.RESULTS)
    rows = [r for r in map(json.loads, open(path)) if r["probe"] == probe]
    if not rows:
        sys.exit(f"no rows for probe={probe} in {path}")
    print(f"\n{path}  probe={probe}  domain={domain}   (% of samples per query)")
    print(f"{'Q':<3}" + "".join(f"{c:>13}" for c in categories) + "   question")
    totals = Counter()
    for qi in sorted({r["question_index"] for r in rows}):
        sub = [r["persona"] for r in rows if r["question_index"] == qi]
        c = Counter(classify(x, domain) for x in sub)
        totals.update(c)
        cells = "".join(f"{100 * c[k] / len(sub):>12.0f}%" for k in categories)
        q = [r["question"] for r in rows if r["question_index"] == qi][0]
        print(f"{qi:<3}{cells}   {q[:40]}")
    n = sum(totals.values())
    print(f"{'ALL':<3}" + "".join(f"{100 * totals[k] / n:>12.0f}%" for k in categories))


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else "interview",
         sys.argv[3] if len(sys.argv) > 3 else "medical")
