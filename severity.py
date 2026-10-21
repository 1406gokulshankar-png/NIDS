# severity.py
# Purpose: turn (attack type + anomaly score) into Low / Medium / High / Critical.

# How dangerous is each kind of attack? (higher = worse)
TYPE_POINTS = {
    "Probe": 1,     # scanning the network, no damage yet
    "DoS": 2,       # flooding a server
    "R2L": 3,       # outsider trying to get in
    "Unknown": 3,   # we don't know what it is, so be careful
    "U2R": 4,       # user trying to become admin (most dangerous)
}


def score_points(score):
    """The more confident / strange the model is, the more points."""
    if score >= 0.9:
        return 2
    if score >= 0.7:
        return 1
    return 0


def compute_severity(attack_type, score):
    """Return 'Low', 'Medium', 'High' or 'Critical' for one alert."""
    # If the attack type is somehow not in our table, treat it like Unknown (3)
    points = TYPE_POINTS.get(attack_type, 3) + score_points(score)

    if points >= 5:
        return "Critical"
    if points == 4:
        return "High"
    if points == 3:
        return "Medium"
    return "Low"


# This block runs only when we type:  python severity.py
# It prints a table so you can see how the rules behave.
if __name__ == "__main__":
    scores = [0.55, 0.75, 0.95]
    print(f"{'attack type':<12}" + "".join(f"score={s:<8}" for s in scores))
    print("-" * 42)
    for attack_type in ["Probe", "DoS", "R2L", "U2R", "Unknown"]:
        row = "".join(f"{compute_severity(attack_type, s):<14}" for s in scores)
        print(f"{attack_type:<12}{row}")