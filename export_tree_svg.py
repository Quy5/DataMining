"""
export_tree_svg.py - Xuất sơ đồ cây ID3 ra SVG (output/tree_svg/) để ứng dụng phóng to/thu nhỏ mà không cần cài Graphviz.
Chạy lại sau khi huấn luyện lại mô hình:  python export_tree_svg.py   (cần Graphviz: lệnh `dot`)
"""
import os
import subprocess

from data_utils import VALID_ATTRIBUTE_VALUES
from id3 import ID3DecisionTree

MODEL_FILE = "output/model.json"
OUT_DIR = "output/tree_svg"
LEAF_FILL = {"unacc": "#FDE2E1", "acc": "#FDEBD3", "good": "#DDF0DE", "vgood": "#DCE6FB"}
CLASS_COLOR = {"unacc": "#B42318", "acc": "#B54708", "good": "#2E7D32", "vgood": "#1D4ED8"}


def tree_dot(model, depth: int, horizontal: bool) -> str:
    """Sơ đồ cây giữ nguyên tên thuộc tính, giá trị và nhãn của car.data (safety, low, unacc…)."""
    lines = [
        "digraph G {",
        f'  rankdir={"LR" if horizontal else "TB"}; nodesep=0.35; ranksep={"0.9" if horizontal else "0.7"}; pad=0.2;',
        '  node [fontname="Helvetica", fontsize=18, shape=box, style="rounded,filled", margin="0.18,0.08",'
        '        color="#94A3B8", fillcolor="#F8FAFC", penwidth=1.6];',
        '  edge [fontname="Helvetica", fontsize=16, color="#64748B", fontcolor="#0F172A", penwidth=1.6, arrowsize=0.8];',
    ]

    def visit(node):
        nid = f"n{node.node_id}"
        if node.is_leaf:
            fill, border = LEAF_FILL[node.prediction], CLASS_COLOR[node.prediction]
            if node.samples_count == 0:
                lines.append(f'  {nid} [label=<<b>{node.prediction}</b><br/><font point-size="14">n = 0 (lá rỗng)</font>>, '
                             f'fillcolor="{fill}", color="{border}", style="rounded,filled,dashed"];')
            else:
                lines.append(f'  {nid} [label=<<b>{node.prediction}</b><br/><font point-size="14">n = {node.samples_count}</font>>, '
                             f'fillcolor="{fill}", color="{border}", penwidth=2.2];')
            return
        if node.depth >= depth:  # nhánh bị thu gọn trên hình vẽ
            lines.append(f'  {nid} [label=<<b>{node.attribute} …</b><br/><font point-size="14">n = {node.samples_count}, '
                         f'đa số {node.majority_class}</font>>, style="rounded,dashed", fillcolor="white"];')
            return
        lines.append(f'  {nid} [label=<<b>{node.attribute}?</b><br/><font point-size="14">n = {node.samples_count}</font>>, '
                     f'color="#1D4ED8", fillcolor="#EFF6FF", penwidth=2.2];')
        order = VALID_ATTRIBUTE_VALUES.get(node.attribute, [])
        for v, child in sorted(node.children.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99):
            lines.append(f'  {nid} -> n{child.node_id} [label=" {v} "];')
            visit(child)

    visit(model.root)
    lines.append("}")
    return "\n".join(lines)


def svg_path(depth: int, horizontal: bool) -> str:
    return os.path.join(OUT_DIR, f"tree_d{depth}_{'ngang' if horizontal else 'doc'}.svg")


def export_all(max_depth: int = 6) -> None:
    model = ID3DecisionTree.load_model(MODEL_FILE)
    os.makedirs(OUT_DIR, exist_ok=True)
    for d in range(1, max_depth + 1):
        for h in (False, True):
            svg = subprocess.run(["dot", "-Tsvg"], input=tree_dot(model, d, h).encode("utf-8"),
                                 capture_output=True, check=True).stdout
            with open(svg_path(d, h), "wb") as f:
                f.write(svg)
            print("Đã ghi", svg_path(d, h))


if __name__ == "__main__":
    export_all()
