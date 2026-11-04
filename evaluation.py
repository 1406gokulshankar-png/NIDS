# evaluation.py
# Purpose: measure how well the replay did, especially on "novel" attacks
# (attacks that appear in the test file but never in the training file).

import json
import os
from datetime import datetime

# Always save next to this file, no matter where we run Python from
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EVALUATION_PATH = os.path.join(BASE_DIR, "evaluation.json")


def percent(part, total):
    """part/total as a percentage. Returns None if total is 0 (avoids dividing by zero)."""
    if total == 0:
        return None
    return round(part / total * 100, 1)


def evaluate(results):
    """Work out the evaluation numbers.

    'results' is a list with ONE dictionary per replayed row, like this:
        {"row_index": 5, "attack_name": "neptune", "true_category": "DoS",
         "is_novel": False, "flagged": True, "alert_type": "DoS"}
    """
    attacks = [r for r in results if r["true_category"] != "Normal"]
    normals = [r for r in results if r["true_category"] == "Normal"]
    known = [r for r in attacks if not r["is_novel"]]    # seen in training
    novel = [r for r in attacks if r["is_novel"]]        # never seen in training

    def detection(rows):
        """How many attacks in this group raised an alert?"""
        detected = sum(1 for r in rows if r["flagged"])
        return {
            "total": len(rows),
            "detected": detected,
            "missed": len(rows) - detected,
            "detection_rate": percent(detected, len(rows)),
        }

    def category_accuracy(rows):
        """Among flagged attacks, how often did the alert type match the TRUE category?"""
        flagged = [r for r in rows if r["flagged"]]
        correct = sum(1 for r in flagged if r["alert_type"] == r["true_category"])
        return {
            "flagged": len(flagged),
            "correct": correct,
            "accuracy": percent(correct, len(flagged)),
        }

    # Detection of each individual novel attack (e.g. mailbomb, apache2...)
    novel_by_attack = {}
    for r in novel:
        entry = novel_by_attack.setdefault(r["attack_name"], {"total": 0, "detected": 0})
        entry["total"] += 1
        if r["flagged"]:
            entry["detected"] += 1

    known_result = detection(known)
    novel_result = detection(novel)
    false_alarms = sum(1 for r in normals if r["flagged"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_replayed": len(results),
        "normal_rows": len(normals),
        "known_attacks": known_result,
        "novel_attacks": novel_result,
        "missed_attacks": known_result["missed"] + novel_result["missed"],  # false negatives
        "false_alarms": false_alarms,                                       # false positives
        "false_alarm_rate": percent(false_alarms, len(normals)),
        "category_accuracy": {
            "known": category_accuracy(known),
            "novel": category_accuracy(novel),
        },
        # For novel attacks, "Unknown" is the designed answer, so we count it separately
        "novel_flagged_as_unknown": sum(
            1 for r in novel if r["flagged"] and r["alert_type"] == "Unknown"),
        "novel_by_attack": novel_by_attack,
    }


def show(rate):
    """Turn 85.3 into '85.3%' and None into 'n/a'."""
    return "n/a" if rate is None else f"{rate}%"


def print_summary(m):
    """Print the evaluation lines (call this inside the REPLAY SUMMARY)."""
    k, n = m["known_attacks"], m["novel_attacks"]
    ck, cn = m["category_accuracy"]["known"], m["category_accuracy"]["novel"]
    print(f"Known attacks detected   : {show(k['detection_rate'])}  ({k['detected']} of {k['total']})")
    print(f"Novel attacks detected   : {show(n['detection_rate'])}  ({n['detected']} of {n['total']})")
    print(f"Missed attacks (false -) : {m['missed_attacks']}  (known {k['missed']}, novel {n['missed']})")
    print(f"False alarms (false +)   : {m['false_alarms']}  "
          f"({show(m['false_alarm_rate'])} of {m['normal_rows']} normal rows)")
    print(f"Category accuracy, known : {show(ck['accuracy'])}  ({ck['correct']} of {ck['flagged']} alerts)")
    print(f"Category accuracy, novel : {show(cn['accuracy'])}  ({cn['correct']} of {cn['flagged']} alerts)")
    print(f"Novel flagged 'Unknown'  : {m['novel_flagged_as_unknown']}")


def save_evaluation(metrics, path=EVALUATION_PATH):
    """Save the numbers to evaluation.json."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_evaluation(path=EVALUATION_PATH):
    """Read evaluation.json. Returns None if the replay hasn't been run yet."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None