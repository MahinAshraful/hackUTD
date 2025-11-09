import pandas as pd
import json

# Load data
df = pd.read_csv("parkinsons_data.csv")

# Average 3 recordings per patient
feature_cols = [
    col for col in df.columns if col not in ["ID", "Recording", "Status", "Gender"]
]
grouped = (
    df.groupby("ID")
    .agg(
        {"Status": "first", "Gender": "first", **{col: "mean" for col in feature_cols}}
    )
    .reset_index()
)

# Split by class
healthy = grouped[grouped["Status"] == 0].sample(frac=1, random_state=42)
pd_patients = grouped[grouped["Status"] == 1].sample(frac=1, random_state=42)

# 75% train, 25% test
n_h_train = int(len(healthy) * 0.75)
n_p_train = int(len(pd_patients) * 0.75)

train = pd.concat([healthy[:n_h_train], pd_patients[:n_p_train]]).sample(
    frac=1, random_state=42
)
test = pd.concat([healthy[n_h_train:], pd_patients[n_p_train:]]).sample(
    frac=1, random_state=42
)

# Save
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

# Save feature stats
stats = {
    col: {"mean": float(train[col].mean()), "std": float(train[col].std())}
    for col in feature_cols
}
with open("feature_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"✓ Train: {len(train)} patients")
print(f"✓ Test: {len(test)} patients")
print("✓ Files saved: train.csv, test.csv, feature_stats.json")
