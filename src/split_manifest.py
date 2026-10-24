import json
import random

manifest_path = "./data/train_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Shuffle and split
random.shuffle(lines)
split_idx = int(0.9 * len(lines))

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open("./data/train_manifest.json", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("./data/val_manifest.json", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Split complete! Train entries: {len(train_lines)}, Validation entries: {len(val_lines)}")
