"""
build_web.py - Tạo bản web tĩnh (index.html) có giao diện giống ứng dụng Streamlit app.py, để đưa lên Vercel.
Mọi số liệu lấy từ output/model.json, output/results.json và data/car.data; dự đoán chạy bằng JavaScript
trên chính cây ID3 đã huấn luyện. Chạy lại sau khi huấn luyện lại mô hình:  python build_web.py
"""
import json

from data_utils import (ATTRIBUTE_NAMES_VI, CLASS_MAPPING_VI, FEATURE_COLUMNS, TARGET_COLUMN,
                        VALID_ATTRIBUTE_VALUES, load_data, split_data)
from id3 import ID3DecisionTree, entropy, information_gain

CLASS_ORDER = ["unacc", "acc", "good", "vgood"]

df = load_data("data/car.data")
train, _ = split_data(df, test_size=0.2, random_state=42, stratify=True)
model = ID3DecisionTree.load_model("output/model.json")
with open("output/model.json", encoding="utf-8") as f:
    tree = json.load(f)["tree"]
with open("output/results.json", encoding="utf-8") as f:
    results = json.load(f)
stats = model.get_tree_stats()

# Cách tính tại nút gốc (trên tập train)
y = train[TARGET_COLUMN]
cnt = y.value_counts().reindex(CLASS_ORDER, fill_value=0)
root = stats["root_attribute"]
gain_rows = []
for a in FEATURE_COLUMNS:
    g, base_ent, _ = information_gain(train, a, TARGET_COLUMN)
    gain_rows.append({"a": a, "infoA": base_ent - g, "gain": g})
gain_rows.sort(key=lambda r: -r["gain"])
parts = []
for v in VALID_ATTRIBUTE_VALUES[root]:
    dj = train[train[root] == v][TARGET_COLUMN]
    c = dj.value_counts().reindex(CLASS_ORDER, fill_value=0)
    parts.append({"v": v, "n": len(dj), **{k: int(c[k]) for k in CLASS_ORDER}, "info": entropy(dj.tolist())})

rules = sorted(
    [{"if": "  VÀ  ".join(f"{a} = {v}" for a, v in r["conditions"]), "then": r["prediction"],
      "n": r["samples"], "k": len(r["conditions"])} for r in model.get_rules()],
    key=lambda r: -r["n"])

DATA = {
    "tree": tree,
    "stats": stats,
    "results": results,
    "rootCalc": {"counts": {k: int(cnt[k]) for k in CLASS_ORDER}, "n": len(y), "infoD": entropy(y.tolist()),
                 "gains": gain_rows, "parts": parts, "root": root},
    "rules": rules,
    "rows": df[FEATURE_COLUMNS + [TARGET_COLUMN]].values.tolist(),
    "attrVi": {a: ATTRIBUTE_NAMES_VI[a] for a in FEATURE_COLUMNS},
    "values": {a: VALID_ATTRIBUTE_VALUES[a] for a in FEATURE_COLUMNS},
    "classVi": {c: CLASS_MAPPING_VI[c]["name"] for c in CLASS_ORDER},
    "classDesc": {c: CLASS_MAPPING_VI[c]["description"] for c in CLASS_ORDER},
}

with open("web_template.html", encoding="utf-8") as f:
    html = f.read().replace("/*__DATA__*/null", json.dumps(DATA, ensure_ascii=False, separators=(",", ":")))
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Đã ghi index.html", len(html) // 1024, "KB")
