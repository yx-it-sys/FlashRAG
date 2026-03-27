#!/usr/bin/env python3
import argparse
from collections import defaultdict
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def excel_col_to_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall("a:si", NS):
        text = "".join(node.text or "" for node in si.iterfind(".//a:t", NS))
        values.append(text)
    return values


def resolve_first_sheet(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    first_sheet = workbook.find("a:sheets/a:sheet", NS)
    rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall("p:Relationship", NS):
        if rel.attrib["Id"] == rid:
            target = rel.attrib["Target"]
            return target if target.startswith("xl/") else f"xl/{target}"
    raise RuntimeError("Cannot resolve first worksheet path")


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//a:t", NS))

    value_node = cell.find("a:v", NS)
    if value_node is None or value_node.text is None:
        return ""

    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    return raw


def load_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_path = resolve_first_sheet(zf)
        sheet = ET.fromstring(zf.read(sheet_path))

    rows = []
    for row in sheet.findall("a:sheetData/a:row", NS):
        values = []
        for cell in row.findall("a:c", NS):
            index = excel_col_to_index(cell.attrib["r"])
            while len(values) <= index:
                values.append("")
            values[index] = cell_value(cell, shared_strings).strip()
        rows.append(values)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xlsx_path",
        nargs="?",
        default="/home/you/papers/literature_update.xlsx",
        help="Path to the xlsx file",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.xlsx_path))
    if not rows:
        raise RuntimeError("Workbook is empty")

    headers = rows[0]
    try:
        year_idx = headers.index("Year")
        retrieval_idx = headers.index("检索模式")
        ada_idx = headers.index("Ada_Type")
    except ValueError as exc:
        raise RuntimeError(f"Required column missing: {exc}") from exc

    denominator = 0
    numerator = 0
    yearly = defaultdict(lambda: {"numerator": 0, "denominator": 0})

    for row in rows[1:]:
        year = row[year_idx].strip() if year_idx < len(row) else ""
        retrieval_mode = row[retrieval_idx].strip() if retrieval_idx < len(row) else ""
        ada_type = row[ada_idx].strip() if ada_idx < len(row) else ""
        qualifies = retrieval_mode == "Agent" or ada_type.startswith("Ada_")
        in_scope = bool(retrieval_mode or ada_type)

        if in_scope:
            denominator += 1
            yearly[year]["denominator"] += 1

        if qualifies:
            numerator += 1
            yearly[year]["numerator"] += 1

    ratio = (numerator / denominator) if denominator else 0.0
    print(f"numerator={numerator}")
    print(f"denominator={denominator}")
    print(f"ratio={ratio:.6f}")
    print(f"percentage={ratio:.2%}")
    print("yearly_breakdown:")
    for year in ["2022", "2023", "2024", "2025"]:
        year_numerator = yearly[year]["numerator"]
        year_denominator = yearly[year]["denominator"]
        year_ratio = (year_numerator / year_denominator) if year_denominator else 0.0
        print(
            f"{year}: numerator={year_numerator}, denominator={year_denominator}, "
            f"ratio={year_ratio:.6f}, percentage={year_ratio:.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
