# test_explain.py
# Purpose: check that SHAP works (explain one attack row and one normal row).

from explain import _load, explain_row

c = _load()
test_df = c["test_df"]

# Find the first row of each kind
attack_index = int(test_df.index[test_df["attack_name"] == "neptune"][0])
normal_index = int(test_df.index[test_df["attack_name"] == "normal"][0])

for title, index in [("ATTACK ROW (neptune)", attack_index), ("NORMAL ROW", normal_index)]:
    result = explain_row(index)
    print("\n" + "=" * 60)
    print(title, "| test row", index, "| true label:", result["true_label"])
    print("=" * 60)
    for f in result["features"]:
        print(f"{f['feature']:<32} value={str(f['value']):<10} "
              f"shap={f['shap']:+.4f}  {f['direction']}")