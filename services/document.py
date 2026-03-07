import os

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

from models.transcript import AnalysisResult


def generate_docx(
    title: str, analysis: AnalysisResult, output_dir: str
) -> str:
    """Generate a .docx document with the analysis results."""
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(11)
    style.font.name = "Calibri"

    # Title
    heading = doc.add_heading(title, level=1)

    # Summary
    doc.add_heading("Саммари", level=2)
    doc.add_paragraph(analysis.summary)

    # Table of contents
    doc.add_heading("Оглавление", level=2)
    for i, section in enumerate(analysis.sections, 1):
        time_range = f"[{section.start_time} - {section.end_time}]"
        doc.add_paragraph(
            f"{i}. {section.title} {time_range}",
            style="List Number",
        )

    # Full transcript by sections
    doc.add_heading("Транскрипт", level=2)
    for section in analysis.sections:
        time_range = f"[{section.start_time} - {section.end_time}]"
        doc.add_heading(f"{section.title} {time_range}", level=3)
        doc.add_paragraph(section.content)

    # Save
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50]
    filename = f"{safe_title}.docx".strip()
    output_path = os.path.join(output_dir, filename)
    doc.save(output_path)

    return output_path
