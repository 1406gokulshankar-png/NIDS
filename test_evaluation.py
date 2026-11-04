# test_evaluation.py
# Run with:  python test_evaluation.py

from evaluation import evaluate


def make_row(i, name, category, novel, flagged, alert_type):
    """Build one fake replay result."""
    return {"row_index": i, "attack_name": name, "true_category": category,
            "is_novel": novel, "flagged": flagged, "alert_type": alert_type}


def test_evaluate_counts():
    results = [
        make_row(0, "neptune",  "DoS",    False, True,  "DoS"),      # known, caught, right type
        make_row(1, "smurf",    "DoS",    False, False, None),       # known, MISSED
        make_row(2, "mailbomb", "DoS",    True,  True,  "Unknown"),  # novel, caught as Unknown
        make_row(3, "apache2",  "DoS",    True,  False, None),       # novel, MISSED
        make_row(4, "normal",   "Normal", False, True,  "DoS"),      # FALSE ALARM
        make_row(5, "normal",   "Normal", False, False, None),       # correctly ignored
    ]
    m = evaluate(results)

    assert m["rows_replayed"] == 6
    assert m["known_attacks"]["detection_rate"] == 50.0     # 1 of 2
    assert m["novel_attacks"]["detection_rate"] == 50.0     # 1 of 2
    assert m["missed_attacks"] == 2
    assert m["false_alarms"] == 1
    assert m["false_alarm_rate"] == 50.0                    # 1 of 2 normal rows
    assert m["category_accuracy"]["known"]["accuracy"] == 100.0   # 1 of 1 flagged
    assert m["category_accuracy"]["novel"]["accuracy"] == 0.0     # "Unknown" != "DoS"
    assert m["novel_flagged_as_unknown"] == 1
    assert m["novel_by_attack"]["mailbomb"] == {"total": 1, "detected": 1}


if __name__ == "__main__":
    test_evaluation_counts = test_evaluate_counts()
    print("test_evaluation: all checks passed")