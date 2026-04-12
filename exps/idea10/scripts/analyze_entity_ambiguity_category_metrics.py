#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from math import hypot
from pathlib import Path
from statistics import mean
from zlib import compress

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment")
REPORT_DIR = Path("/home/you/FlashRAG/exps/idea10/idea_reports")
INTERMEDIATE_PATH = BASE_DIR / "intermediate_data.json"
TRAJECTORY_PATH = BASE_DIR / "omnisearch_trajectories.jsonl"
LABEL_PATH = BASE_DIR / "label/qwen/trajectory_annotation.llm_labeled.adjudicated.jsonl"

JSON_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_category_metrics.json"
CSV_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_category_metrics.csv"
MD_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_category_metrics.md"
HEATMAP_SVG_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_accuracy_heatmap.svg"
HEATMAP_PDF_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_accuracy_heatmap.pdf"
RADAR_ORIGIN_CSV_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_accuracy_radar_origin.csv"
ITERATION_SVG_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_iteration_bar.svg"
ITERATION_PDF_OUTPUT_PATH = REPORT_DIR / "qwen/entity_ambiguity_iteration_bar.pdf"

CATEGORY_ORDER = [
    "No",
    "Object Identification",
    "Indirect Entity Ambiguity",
    "Description",
    "No Object Involved",
]
METRIC_ORDER = ["gpt_acc", "f1", "recall", "precision", "rouge-l"]
RADAR_METRIC_ORDER = ["gpt_acc", "f1", "recall", "precision", "rouge-l"]
METRIC_LABELS = {
    "gpt_acc": "LLM-Judge",
    "f1": "F1",
    "recall": "Recall",
    "precision": "Precision",
    "rouge-l": "ROUGE-L",
}
AMBIGUOUS_BAR_COLOR = "#2F6C8F"
NO_BAR_COLOR = "#6C757D"
GRID_COLOR = "#D9DEE5"
TEXT_DARK = "#222222"
TEXT_MID = "#5B6573"
RADAR_RING_LABEL_FONT_SIZE = 14
RADAR_AXIS_LABEL_FONT_SIZE = 19
RADAR_LEGEND_FONT_SIZE = 16
RADAR_LEGEND_ROW_GAP = 34


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def category_from_label(llm_label: dict | None) -> str | None:
    if not isinstance(llm_label, dict):
        return None
    if llm_label.get("entity_ambiguous") == "No":
        return "No"
    category = llm_label.get("ambiguity_level")
    if category in CATEGORY_ORDER:
        return category
    return None


def build_sample_categories() -> dict[str, dict]:
    category_map = {}
    filtered_multi_category_ids = []
    for item in read_jsonl(LABEL_PATH):
        annotation_id = item["annotation_id"]
        query_steps = item.get("query_steps", [])
        first_category = None
        category_sequence = []
        unique_non_no = set()

        for query_step in query_steps:
            category = category_from_label(query_step.get("llm_label"))
            if category is None:
                continue
            category_sequence.append(category)
            if first_category is None:
                first_category = category
            if category != "No":
                unique_non_no.add(category)

        if len(unique_non_no) > 1:
            filtered_multi_category_ids.append(annotation_id)
            continue

        category_map[annotation_id] = {
            "category": first_category or "No",
            "query_count": item.get("query_count", 0),
            "label_sequence": category_sequence,
        }

    return {
        "category_map": category_map,
        "filtered_multi_category_ids": filtered_multi_category_ids,
    }


def build_iteration_counts() -> dict[str, int]:
    counts = {}
    for item in read_jsonl(TRAJECTORY_PATH):
        sample_id = item.get("id")
        if not sample_id:
            continue
        rounds = sum(
            1
            for step in item.get("trajectory", [])
            if step.get("action") in {
                "text_retrieval_result",
                "image_retrieval_result",
                "no_retrieval_result",
            }
        )
        counts[sample_id] = rounds
    return counts


def compute_stats() -> dict:
    intermediate_items = json.loads(INTERMEDIATE_PATH.read_text(encoding="utf-8"))
    label_info = build_sample_categories()
    category_map = label_info["category_map"]
    iteration_counts = build_iteration_counts()
    filtered_multi_category_ids = set(label_info["filtered_multi_category_ids"])

    grouped = defaultdict(list)
    missing_category_ids = []
    missing_iteration_ids = []
    filtered_multi_category_samples = []

    for item in intermediate_items:
        sample_id = item.get("id") or item.get("data_id")
        if sample_id in filtered_multi_category_ids:
            filtered_multi_category_samples.append(sample_id)
            continue
        category_meta = category_map.get(sample_id)
        if category_meta is None:
            missing_category_ids.append(sample_id)
            continue

        metrics = item.get("output", {}).get("metric_score", {})
        if sample_id not in iteration_counts:
            missing_iteration_ids.append(sample_id)
            continue

        grouped[category_meta["category"]].append(
            {
                "id": sample_id,
                "metrics": {metric: float(metrics.get(metric, 0.0)) for metric in METRIC_ORDER},
                "iterations": int(iteration_counts[sample_id]),
                "query_count": int(category_meta.get("query_count", 0)),
            }
        )

    category_rows = []
    for category in CATEGORY_ORDER:
        rows = grouped.get(category, [])
        if rows:
            metric_means = {
                metric: mean(row["metrics"][metric] for row in rows)
                for metric in METRIC_ORDER
            }
            avg_iterations = mean(row["iterations"] for row in rows)
            avg_query_count = mean(row["query_count"] for row in rows)
        else:
            metric_means = {metric: 0.0 for metric in METRIC_ORDER}
            avg_iterations = 0.0
            avg_query_count = 0.0

        category_rows.append(
            {
                "category": category,
                "count": len(rows),
                "avg_iterations": avg_iterations,
                "avg_query_count": avg_query_count,
                "metrics": metric_means,
            }
        )

    return {
        "meta": {
            "num_samples": len(intermediate_items),
            "num_retained_samples": sum(len(rows) for rows in grouped.values()),
            "missing_category_ids": missing_category_ids,
            "missing_iteration_ids": missing_iteration_ids,
            "filtered_multi_category_sample_count": len(filtered_multi_category_samples),
            "filtered_multi_category_sample_ids": filtered_multi_category_samples,
            "category_rule": "Exclude samples containing more than one non-No entity ambiguity category across labeled query steps; otherwise, use the first LLM-labeled text-retrieval query category in the sample. If no labeled text query exists, assign No.",
            "iteration_rule": "Count retrieval-result turns in omnisearch_trajectories.jsonl, i.e. the number of text_retrieval_result, image_retrieval_result, and no_retrieval_result nodes.",
        },
        "categories": category_rows,
    }


def write_outputs(stats: dict) -> None:
    JSON_OUTPUT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "avg_iterations", "avg_query_count", *METRIC_ORDER])
        for row in stats["categories"]:
            writer.writerow(
                [
                    row["category"],
                    row["count"],
                    f"{row['avg_iterations']:.4f}",
                    f"{row['avg_query_count']:.4f}",
                    *[f"{row['metrics'][metric]:.4f}" for metric in METRIC_ORDER],
                ]
            )

    md_lines = [
        "# Entity Ambiguity Category Metrics",
        "",
        f"- Samples: {stats['meta']['num_samples']}",
        f"- Retained samples: {stats['meta']['num_retained_samples']}",
        f"- Filtered multi-category samples: {stats['meta']['filtered_multi_category_sample_count']}",
        f"- Category rule: {stats['meta']['category_rule']}",
        f"- Iteration rule: {stats['meta']['iteration_rule']}",
        "",
        "| Category | Count | Avg Iterations | Avg Query Count | GPTAcc | F1 | Recall | Precision | ROUGE-L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["categories"]:
        md_lines.append(
            "| {category} | {count} | {avg_iterations:.3f} | {avg_query_count:.3f} | {gpt_acc:.3f} | {f1:.3f} | {recall:.3f} | {precision:.3f} | {rougel:.3f} |".format(
                category=row["category"],
                count=row["count"],
                avg_iterations=row["avg_iterations"],
                avg_query_count=row["avg_query_count"],
                gpt_acc=row["metrics"]["gpt_acc"],
                f1=row["metrics"]["f1"],
                recall=row["metrics"]["recall"],
                precision=row["metrics"]["precision"],
                rougel=row["metrics"]["rouge-l"],
            )
        )
    MD_OUTPUT_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with RADAR_ORIGIN_CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "gpt_acc", "f1", "recall", "precision", "rouge-l"])
        for row in stats["categories"]:
            writer.writerow(
                [
                    row["category"],
                    f"{row['metrics']['gpt_acc']:.6f}",
                    f"{row['metrics']['f1']:.6f}",
                    f"{row['metrics']['recall']:.6f}",
                    f"{row['metrics']['precision']:.6f}",
                    f"{row['metrics']['rouge-l']:.6f}",
                ]
            )


def draw_dashed_line(draw, start, end, fill, width, pattern):
    x1, y1 = start
    x2, y2 = end
    total_length = hypot(x2 - x1, y2 - y1)
    if total_length == 0:
        return
    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length
    distance = 0.0
    pattern_idx = 0
    draw_on = True
    while distance < total_length:
        seg = pattern[pattern_idx % len(pattern)]
        next_distance = min(distance + seg, total_length)
        if draw_on:
            sx = x1 + dx * distance
            sy = y1 + dy * distance
            ex = x1 + dx * next_distance
            ey = y1 + dy * next_distance
            draw.line((sx, sy, ex, ey), fill=fill, width=width)
        distance = next_distance
        pattern_idx += 1
        draw_on = not draw_on


def draw_accuracy_heatmap_svg(stats: dict) -> None:
    from math import cos, pi, sin

    width, height = 980, 560
    cx, cy = 345, 285
    radius = 190
    radar_max = 0.5
    colors = {
        "No": "#E64B35",
        "Object Identification": "#F39B45",
        "Indirect Entity Ambiguity": "#F5C04E",
        "Description": "#B9D9B7",
        "No Object Involved": "#6EA6D7",
    }
    angles = [-pi / 2 + 2 * pi * idx / len(RADAR_METRIC_ORDER) for idx in range(len(RADAR_METRIC_ORDER))]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for level in range(1, 6):
        r = radius * level / 5
        if level == 5:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="#444444" stroke-width="1.2"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="#D7DDE5" stroke-width="1" stroke-dasharray="2 2"/>')
        parts.append(
            f'<text x="{cx + 10}" y="{cy - r + 5:.1f}" font-size="{RADAR_RING_LABEL_FONT_SIZE}" font-family="Arial, Helvetica, sans-serif" fill="#6B7280">{radar_max * level / 5:.1f}</text>'
        )

    for angle, metric in zip(angles, RADAR_METRIC_ORDER):
        x = cx + radius * cos(angle)
        y = cy + radius * sin(angle)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#D7DDE5" stroke-width="1.1"/>')
        label = METRIC_LABELS[metric]
        extra_offset = 48 if metric == "rouge-l" else 38
        lx = cx + (radius + extra_offset) * cos(angle)
        ly = cy + (radius + extra_offset) * sin(angle)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="{RADAR_AXIS_LABEL_FONT_SIZE}" font-family="Arial, Helvetica, sans-serif" fill="#222222">{label}</text>'
        )

    for row in stats["categories"]:
        points = []
        point_xy = []
        for angle, metric in zip(angles, RADAR_METRIC_ORDER):
            value = min(row["metrics"][metric], radar_max) / radar_max
            px = cx + radius * value * cos(angle)
            py = cy + radius * value * sin(angle)
            points.append(f"{px:.1f},{py:.1f}")
            point_xy.append((px, py))
        color = colors[row["category"]]
        parts.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for px, py in point_xy:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="{color}" stroke="{color}" stroke-width="1"/>')

    legend_x = 560
    legend_y = 58
    for idx, row in enumerate(stats["categories"]):
        y = legend_y + idx * RADAR_LEGEND_ROW_GAP
        color = colors[row["category"]]
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{legend_x + 13}" cy="{y}" r="3" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 38}" y="{y + 6}" font-size="{RADAR_LEGEND_FONT_SIZE}" font-family="Arial, Helvetica, sans-serif" fill="#222222">{row["category"]}</text>'
        )

    parts.append("</svg>")
    HEATMAP_SVG_OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")


def draw_accuracy_heatmap_pdf(stats: dict) -> None:
    from math import cos, pi, sin

    width, height = 980, 560
    cx, cy = 345, 285
    radius = 190
    radar_max = 0.5
    colors = {
        "No": "#E64B35",
        "Object Identification": "#F39B45",
        "Indirect Entity Ambiguity": "#F5C04E",
        "Description": "#B9D9B7",
        "No Object Involved": "#6EA6D7",
    }
    angles = [-pi / 2 + 2 * pi * idx / len(RADAR_METRIC_ORDER) for idx in range(len(RADAR_METRIC_ORDER))]

    cmds = []
    for level in range(1, 6):
        r = radius * level / 5
        if level == 5:
            cmds.extend(pdf_circle(cx, cy, r, width, height, "#444444", line_width=1.0))
        else:
            cmds.extend(pdf_circle(cx, cy, r, width, height, "#D7DDE5", line_width=0.8, dash=[2, 2]))
        cmds.extend(pdf_text(cx + 10, cy - r + 5, f"{radar_max * level / 5:.1f}", width, height, RADAR_RING_LABEL_FONT_SIZE, "#6B7280"))

    for angle, metric in zip(angles, RADAR_METRIC_ORDER):
        x = cx + radius * cos(angle)
        y = cy + radius * sin(angle)
        cmds.extend(pdf_line(cx, cy, x, y, width, height, "#D7DDE5", line_width=0.9))
        label = METRIC_LABELS[metric]
        extra_offset = 48 if metric == "rouge-l" else 38
        lx = cx + (radius + extra_offset) * cos(angle)
        ly = cy + (radius + extra_offset) * sin(angle)
        cmds.extend(pdf_text(lx - 24, ly + 6, label, width, height, RADAR_AXIS_LABEL_FONT_SIZE, "#222222"))

    for row in stats["categories"]:
        pts = []
        for angle, metric in zip(angles, RADAR_METRIC_ORDER):
            value = min(row["metrics"][metric], radar_max) / radar_max
            pts.append((cx + radius * value * cos(angle), cy + radius * value * sin(angle)))
        cmds.extend(pdf_polyline(pts + [pts[0]], width, height, colors[row["category"]], line_width=1.8))
        for px, py in pts:
            cmds.extend(pdf_filled_circle(px, py, 2.6, width, height, colors[row["category"]]))

    legend_x = 560
    legend_y = 58
    for idx, row in enumerate(stats["categories"]):
        y = legend_y + idx * RADAR_LEGEND_ROW_GAP
        color = colors[row["category"]]
        cmds.extend(pdf_line(legend_x, y, legend_x + 26, y, width, height, color, line_width=1.8))
        cmds.extend(pdf_filled_circle(legend_x + 13, y, 2.6, width, height, color))
        cmds.extend(pdf_text(legend_x + 38, y + 5, row["category"], width, height, RADAR_LEGEND_FONT_SIZE, "#222222"))

    write_vector_pdf(HEATMAP_PDF_OUTPUT_PATH, width, height, cmds)


def hex_to_rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def pdf_y(y: float, page_height: float) -> float:
    return page_height - y


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_line(x1, y1, x2, y2, page_width, page_height, color, line_width=1.0, dash=None):
    r, g, b = hex_to_rgb(color)
    dash_cmd = "[] 0 d" if not dash else f"[{' '.join(f'{v:.2f}' for v in dash)}] 0 d"
    return [
        f"{dash_cmd}\n{line_width:.2f} w\n{r:.4f} {g:.4f} {b:.4f} RG\n{x1:.2f} {pdf_y(y1, page_height):.2f} m\n{x2:.2f} {pdf_y(y2, page_height):.2f} l\nS\n"
    ]


def pdf_polyline(points, page_width, page_height, color, line_width=1.0):
    r, g, b = hex_to_rgb(color)
    cmds = [f"[] 0 d\n{line_width:.2f} w\n{r:.4f} {g:.4f} {b:.4f} RG\n"]
    first = True
    for x, y in points:
        if first:
            cmds.append(f"{x:.2f} {pdf_y(y, page_height):.2f} m\n")
            first = False
        else:
            cmds.append(f"{x:.2f} {pdf_y(y, page_height):.2f} l\n")
    cmds.append("S\n")
    return ["".join(cmds)]


def pdf_circle(cx, cy, r, page_width, page_height, color, line_width=1.0, dash=None):
    k = 0.552284749831
    r_col, g_col, b_col = hex_to_rgb(color)
    dash_cmd = "[] 0 d" if not dash else f"[{' '.join(f'{v:.2f}' for v in dash)}] 0 d"
    y0 = pdf_y(cy, page_height)
    return [
        (
            f"{dash_cmd}\n{line_width:.2f} w\n{r_col:.4f} {g_col:.4f} {b_col:.4f} RG\n"
            f"{cx + r:.2f} {y0:.2f} m\n"
            f"{cx + r:.2f} {y0 - k*r:.2f} {cx + k*r:.2f} {y0 - r:.2f} {cx:.2f} {y0 - r:.2f} c\n"
            f"{cx - k*r:.2f} {y0 - r:.2f} {cx - r:.2f} {y0 - k*r:.2f} {cx - r:.2f} {y0:.2f} c\n"
            f"{cx - r:.2f} {y0 + k*r:.2f} {cx - k*r:.2f} {y0 + r:.2f} {cx:.2f} {y0 + r:.2f} c\n"
            f"{cx + k*r:.2f} {y0 + r:.2f} {cx + r:.2f} {y0 + k*r:.2f} {cx + r:.2f} {y0:.2f} c\nS\n"
        )
    ]


def pdf_filled_circle(cx, cy, r, page_width, page_height, color):
    k = 0.552284749831
    r_col, g_col, b_col = hex_to_rgb(color)
    y0 = pdf_y(cy, page_height)
    return [
        (
            f"{r_col:.4f} {g_col:.4f} {b_col:.4f} rg\n"
            f"{cx + r:.2f} {y0:.2f} m\n"
            f"{cx + r:.2f} {y0 - k*r:.2f} {cx + k*r:.2f} {y0 - r:.2f} {cx:.2f} {y0 - r:.2f} c\n"
            f"{cx - k*r:.2f} {y0 - r:.2f} {cx - r:.2f} {y0 - k*r:.2f} {cx - r:.2f} {y0:.2f} c\n"
            f"{cx - r:.2f} {y0 + k*r:.2f} {cx - k*r:.2f} {y0 + r:.2f} {cx:.2f} {y0 + r:.2f} c\n"
            f"{cx + k*r:.2f} {y0 + r:.2f} {cx + r:.2f} {y0 + k*r:.2f} {cx + r:.2f} {y0:.2f} c\nf\n"
        )
    ]


def pdf_text(x, y, text, page_width, page_height, size, color):
    r, g, b = hex_to_rgb(color)
    return [
        f"BT\n/F1 {size:.2f} Tf\n{r:.4f} {g:.4f} {b:.4f} rg\n1 0 0 1 {x:.2f} {pdf_y(y, page_height):.2f} Tm\n({pdf_escape(text)}) Tj\nET\n"
    ]


def write_vector_pdf(pdf_path: Path, width: int, height: int, content_commands: list[str]) -> None:
    stream = "".join(content_commands).encode("latin-1")
    compressed_stream = compress(stream)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(compressed_stream)} /Filter /FlateDecode >>\nstream\n".encode("latin-1") + compressed_stream + b"\nendstream",
    ]
    xref = []
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    for idx, obj in enumerate(objects, start=1):
        xref.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        if isinstance(obj, str):
            pdf.extend(obj.encode("latin-1"))
        else:
            pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in xref:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("latin-1"))
    pdf_path.write_bytes(pdf)


def draw_iteration_bar_svg(stats: dict) -> None:
    rows = sorted(stats["categories"], key=lambda row: row["avg_iterations"], reverse=True)
    width, height = 1120, 620
    left = 360
    top = 95
    plot_width = 620
    bar_height = 42
    gap = 18
    max_value = max(max(row["avg_iterations"] for row in rows), 1.0)
    colors = ["#F4A261", "#F6C177", "#FFD089", "#CFE8F3", "#A8D0E6"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top - 28}" width="{plot_width}" height="{len(rows) * (bar_height + gap) - gap + 50}" fill="none" stroke="#444444" stroke-width="1.4"/>',
    ]

    defs = ['<defs>']
    for idx, color in enumerate(colors):
        defs.append(
            f'<pattern id="diag{idx}" patternUnits="userSpaceOnUse" width="14" height="14" patternTransform="rotate(45)">'
            f'<rect width="14" height="14" fill="{color}"/>'
            f'<line x1="0" y1="0" x2="0" y2="14" stroke="#666666" stroke-width="3"/>'
            f'</pattern>'
        )
    defs.append('</defs>')
    parts.extend(defs)

    for tick_idx in range(6):
        tick_value = max_value * tick_idx / 5
        x = left + plot_width * tick_idx / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top - 28}" x2="{x:.1f}" y2="{top + len(rows) * (bar_height + gap) - gap + 22}" stroke="#D9DEE5" stroke-width="1" stroke-dasharray="3 3"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + len(rows) * (bar_height + gap) + 32}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#222222">{tick_value:.1f}</text>'
        )

    for idx, row in enumerate(rows):
        y = top + idx * (bar_height + gap)
        fill_width = plot_width * row["avg_iterations"] / max_value if max_value else 0
        parts.append(
            f'<text x="{left - 14}" y="{y + 28}" text-anchor="end" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#222222">{row["category"]}</text>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 46}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#6B7280">n={row["count"]}</text>'
        )
        parts.append(f'<rect x="{left}" y="{y}" width="{fill_width:.1f}" height="{bar_height}" fill="url(#diag{idx % len(colors)})" stroke="#666666" stroke-width="1"/>')
        parts.append(
            f'<text x="{left + fill_width + 10:.1f}" y="{y + 28}" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#222222">{row["avg_iterations"]:.2f}</text>'
        )

    parts.append(
        f'<text x="{left + plot_width / 2}" y="{top + len(rows) * (bar_height + gap) + 60}" text-anchor="middle" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#222222">Average Search Rounds</text>'
    )
    parts.append("</svg>")
    ITERATION_SVG_OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")


def draw_iteration_bar_pdf(stats: dict) -> None:
    rows = sorted(stats["categories"], key=lambda row: row["avg_iterations"], reverse=True)
    width, height = 1120, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    text_font = load_font(16)
    small_font = load_font(14)

    left = 360
    top = 95
    plot_width = 620
    bar_height = 42
    gap = 18
    max_value = max(max(row["avg_iterations"] for row in rows), 1.0)
    plot_bottom = top + len(rows) * (bar_height + gap) - gap + 22
    plot_top = top - 28
    colors = ["#F4A261", "#F6C177", "#FFD089", "#CFE8F3", "#A8D0E6"]

    draw.rectangle((left, plot_top, left + plot_width, plot_bottom), outline="#444444", width=2)

    for tick_idx in range(6):
        tick_value = max_value * tick_idx / 5
        x = left + plot_width * tick_idx / 5
        for yy in range(plot_top, plot_bottom, 8):
            draw.line((x, yy, x, min(yy + 4, plot_bottom)), fill="#D9DEE5", width=1)
        tick_text = f"{tick_value:.1f}"
        tick_box = draw.textbbox((0, 0), tick_text, font=small_font)
        draw.text((x - (tick_box[2] - tick_box[0]) / 2, plot_bottom + 10), tick_text, fill=TEXT_MID, font=small_font)

    for idx, row in enumerate(rows):
        y = top + idx * (bar_height + gap)
        fill_width = plot_width * row["avg_iterations"] / max_value if max_value else 0
        label_box = draw.textbbox((0, 0), row["category"], font=text_font)
        draw.text((left - 14 - (label_box[2] - label_box[0]), y + 8), row["category"], fill=TEXT_DARK, font=text_font)
        n_text = f"n={row['count']}"
        n_box = draw.textbbox((0, 0), n_text, font=small_font)
        draw.text((left - 14 - (n_box[2] - n_box[0]), y + 27), n_text, fill=TEXT_MID, font=small_font)
        draw.rectangle((left, y, left + fill_width, y + bar_height), fill=colors[idx % len(colors)], outline="#666666", width=1)
        step = 12
        x0 = left - bar_height
        while x0 < left + fill_width:
            draw.line((x0, y + bar_height, x0 + bar_height, y), fill="#666666", width=2)
            x0 += step
        draw.text((left + fill_width + 8, y + 8), f"{row['avg_iterations']:.2f}", fill=TEXT_DARK, font=text_font)

    axis_title = "Average Search Rounds"
    axis_box = draw.textbbox((0, 0), axis_title, font=small_font)
    draw.text((left + plot_width / 2 - (axis_box[2] - axis_box[0]) / 2, plot_bottom + 38), axis_title, fill=TEXT_MID, font=small_font)

    image.save(ITERATION_PDF_OUTPUT_PATH, "PDF", resolution=300.0)


def main() -> None:
    stats = compute_stats()
    write_outputs(stats)
    draw_accuracy_heatmap_pdf(stats)
    draw_iteration_bar_pdf(stats)
    print(f"Saved {JSON_OUTPUT_PATH}")
    print(f"Saved {CSV_OUTPUT_PATH}")
    print(f"Saved {MD_OUTPUT_PATH}")
    print(f"Saved {HEATMAP_SVG_OUTPUT_PATH}")
    print(f"Saved {HEATMAP_PDF_OUTPUT_PATH}")
    print(f"Saved {RADAR_ORIGIN_CSV_OUTPUT_PATH}")
    print(f"Saved {ITERATION_SVG_OUTPUT_PATH}")
    print(f"Saved {ITERATION_PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
