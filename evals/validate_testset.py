import json
from pathlib import Path

path = Path("evals/testset/test_cases.jsonl")

required_top = {"id", "category", "prompt", "expected"}
required_expected = {
    "should_say_idk",
    "citation_required",
    "must_include",
    "must_not_include",
    "preferred_sources",
}

allowed_categories = {"normal", "edge", "idk"}

if not path.exists():
    raise SystemExit(f"❌ Missing file: {path}")

bad = []
ids = set()
count = 0

for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue

    count += 1
    try:
        obj = json.loads(line)
    except Exception as e:
        bad.append((i, f"Invalid JSON: {e}"))
        continue

    missing = required_top - obj.keys()
    if missing:
        bad.append((i, f"Missing top fields: {missing}"))
        continue

    if obj["id"] in ids:
        bad.append((i, f"Duplicate id: {obj['id']}"))
    ids.add(obj["id"])

    if obj["category"] not in allowed_categories:
        bad.append((i, f"Invalid category: {obj['category']}"))

    exp = obj["expected"]
    missing_exp = required_expected - exp.keys()
    if missing_exp:
        bad.append((i, f"Missing expected fields: {missing_exp}"))

print(f"Checked {count} lines ({len(ids)} unique ids).")
if bad:
    print("❌ Errors:")
    for row in bad:
        print(" -", row)
    raise SystemExit(1)

print("✅ Test set looks valid.")
