import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "annotated_failure_cases.json"
SVG_PATH = BASE_DIR / "category_distribution_bar.svg"
PDF_PATH = BASE_DIR / "category_distribution_bar.pdf"

ALL_CATEGORIES = [
    "Explicit Deictics",
    "Attribute-as-Name",
    "Spatial Dependency",
    "Category-level Vagueness",
    "Visual Recognition Task Offloading",
]

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_svg(labels, values, total):
    width, height = 1200, 760
    left, right, top, bottom = 320, 80, 120, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(max(values), 1)
    bar_gap = 20
    bar_height = (plot_height - bar_gap * (len(labels) - 1)) / len(labels)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="52" text-anchor="middle" font-size="28" font-family="Arial, Helvetica, sans-serif" font-weight="bold">',
        "Category Distribution of Annotated Failure Cases",
        "</text>",
    ]

    for tick in range(0, max_value + 1, 5):
        x = left + plot_width * tick / max_value
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 36}" text-anchor="middle" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#444444">{tick}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444444" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#444444" stroke-width="1.5"/>')

    for idx, (label, value, color) in enumerate(zip(labels, values, COLORS)):
        y = top + idx * (bar_height + bar_gap)
        bar_width = plot_width * value / max_value if max_value else 0
        pct = 100.0 * value / total if total else 0.0
        cy = y + bar_height / 2

        parts.append(
            f'<text x="{left - 16}" y="{cy + 6:.1f}" text-anchor="end" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="black">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="6" ry="6" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 12:.1f}" y="{cy + 6:.1f}" text-anchor="start" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="black">{value} ({pct:.1f}%)</text>'
        )

    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 28}" text-anchor="middle" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="black">Number of Cases</text>'
    )
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts))


def draw_pdf(labels, values, total):
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    label_font = load_font(18)
    value_font = load_font(18)
    axis_font = load_font(18)

    left, right, top, bottom = 320, 80, 120, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(max(values), 1)
    bar_gap = 20
    bar_height = (plot_height - bar_gap * (len(labels) - 1)) / len(labels)

    title = "Category Distribution of Annotated Failure Cases"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 20), title, fill="black", font=title_font)

    for tick in range(0, max_value + 1, 5):
        x = left + plot_width * tick / max_value
        draw.line((x, top, x, height - bottom), fill="#e6e6e6", width=1)
        tick_text = str(tick)
        tick_box = draw.textbbox((0, 0), tick_text, font=axis_font)
        draw.text((x - (tick_box[2] - tick_box[0]) / 2, height - bottom + 12), tick_text, fill="#444444", font=axis_font)

    draw.line((left, top, left, height - bottom), fill="#444444", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#444444", width=2)

    for idx, (label, value, color) in enumerate(zip(labels, values, COLORS)):
        y = top + idx * (bar_height + bar_gap)
        bar_width = plot_width * value / max_value if max_value else 0
        pct = 100.0 * value / total if total else 0.0
        cy = y + bar_height / 2

        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 16 - (label_box[2] - label_box[0]), cy - (label_box[3] - label_box[1]) / 2), label, fill="black", font=label_font)
        draw.rounded_rectangle((left, y, left + bar_width, y + bar_height), radius=6, fill=color)

        value_text = f"{value} ({pct:.1f}%)"
        draw.text((left + bar_width + 12, cy - 10), value_text, fill="black", font=value_font)

    axis_label = "Number of Cases"
    axis_box = draw.textbbox((0, 0), axis_label, font=axis_font)
    draw.text((left + plot_width / 2 - (axis_box[2] - axis_box[0]) / 2, height - 48), axis_label, fill="black", font=axis_font)

    image.save(PDF_PATH, "PDF", resolution=300.0)


def main():
    data = json.loads(INPUT_PATH.read_text())
    counts = Counter(item["category"] for item in data["annotations"])
    labels = ALL_CATEGORIES
    values = [counts[label] for label in labels]
    total = sum(values)

    draw_svg(labels, values, total)
    draw_pdf(labels, values, total)

    print(f"Saved {SVG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
