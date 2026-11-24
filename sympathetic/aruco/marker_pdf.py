# marker_pdf.py developed by Bob Tianqi Wei

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


PAPER_SIZES = {"US Letter": LETTER, "A4": A4}


def layout_markers(markers, page_size, dictionary):
    margin, gap = 12 * mm, 3 * mm
    width, height = page_size
    cells = dictionary.markerSize + 2
    pages = [[]]
    left, top, row_height = margin, height - margin, 0

    for index, (id_text, part, size_text) in enumerate(markers, start=1):
        try:
            marker_id = int(id_text)
            size_mm = float(size_text)
        except ValueError:
            raise ValueError(f"Row {index}: enter a whole-number ID and a size in mm.") from None
        if not 0 <= marker_id < len(dictionary.bytesList):
            raise ValueError(f"Row {index}: ID must be from 0 to {len(dictionary.bytesList) - 1}.")
        if not math.isfinite(size_mm) or size_mm <= 0:
            raise ValueError(f"Row {index}: size must be greater than 0 mm.")

        size = size_mm * mm
        quiet_zone = size / cells
        label = f"{marker_id} - {part}"
        footprint = size + 2 * quiet_zone
        item_width = max(footprint, stringWidth(label, "Helvetica", 9))
        item_height = footprint + 14
        if item_width > width - 2 * margin or item_height > height - 2 * margin:
            raise ValueError(f"Row {index}: marker and white margin do not fit on this paper.")

        if left + item_width > width - margin:
            left, top, row_height = margin, top - row_height - gap, 0
        if top - item_height < margin:
            pages.append([])
            left, top, row_height = margin, height - margin, 0

        center_x = left + item_width / 2
        marker_bottom = top - quiet_zone - size
        pages[-1].append((marker_id, size, center_x, marker_bottom, label, top - footprint - 10))
        left += item_width + gap
        row_height = max(row_height, item_height)

    if not pages[0]:
        raise ValueError("Add at least one marker.")
    return pages


def write_pdf(path, page_size, dictionary, pages):
    cells = dictionary.markerSize + 2
    with TemporaryDirectory(dir=path.parent) as folder:
        temporary_pdf = Path(folder) / "markers.pdf"
        pdf = Canvas(str(temporary_pdf), pagesize=page_size, pdfVersion=(1, 7))
        pdf.setTitle("ArUco Markers")
        pdf.setViewerPreference("PrintScaling", "None")
        for page in pages:
            pdf.setFont("Helvetica", 9)
            for marker_id, size, center_x, bottom, label, label_y in page:
                left = center_x - size / 2
                cell_size = size / cells
                grid = cv2.aruco.generateImageMarker(dictionary, marker_id, cells)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.rect(left, bottom, size, size, stroke=0, fill=1)
                white_cells = pdf.beginPath()
                for row, col in zip(*grid.nonzero()):
                    x = left + col * cell_size
                    y = bottom + (cells - row - 1) * cell_size
                    white_cells.rect(x, y, cell_size, cell_size)
                pdf.setFillColorRGB(1, 1, 1)
                pdf.drawPath(white_cells, stroke=0, fill=1)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.drawCentredString(center_x, label_y, label)
            pdf.showPage()
        pdf.save()
        temporary_pdf.replace(path)
