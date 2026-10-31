#!/usr/bin/env python3
"""Preserve the source network drawing and enlarge its legend for one column.

The road and route geometry are reused unchanged from the project manuscript.
Only the original legend is covered and replaced with larger vector lettering.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tits-network-mpl"))
import fitz
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
source = fitz.open(ROOT / "figures" / "network_topology_original.pdf")
sp = source[0]
width = 252.0
scale = width / sp.rect.width
map_height = sp.rect.height * scale
result = fitz.open()
page = result.new_page(width=width, height=map_height + 51.0)
font_path = font_manager.findfont("DejaVu Sans")
page.insert_font(fontname="legend", fontfile=font_path)
page.show_pdf_page(fitz.Rect(0, 0, width, map_height), source, 0)

# The original legend is an opaque white box in the lower left of the source.
# Cover only that box; the route drawing and map scale are unchanged.
old_legend = fitz.Rect(10.5, 544.3, 174.2, 630.2)
page.draw_rect(old_legend * fitz.Matrix(scale, scale),
               color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
labels = ["7X", "7S", "102X", "102S", "311X", "311S",
          "122X", "122S", "406X", "406S", "705X", "705S"]
colors = [
    (0.8392157, 0.1529412, 0.1568627), (1, 0.4980392, 0.0549020),
    (0.1725490, 0.6274510, 0.1725490), (0.5960785, 0.8745098, 0.5411765),
    (0.1215686, 0.4666667, 0.7058824), (0.6823530, 0.7803922, 0.9098040),
    (0.5803922, 0.4039216, 0.7411765), (0.7725490, 0.6901961, 0.8352941),
    (0.5490196, 0.3372549, 0.2941177), (0.7686275, 0.6117647, 0.5803922),
    (0.8901961, 0.4666667, 0.7607843), (0.9686275, 0.7137255, 0.8235294),
]
page.insert_text((3, map_height + 7), "Directional services", fontsize=8,
                 fontname="legend", color=(0.1, 0.1, 0.1))
for i, (label, color) in enumerate(zip(labels, colors)):
    row, col = divmod(i, 4)
    x, y = 4 + col * 62, map_height + 17 + row * 10
    page.draw_line((x, y-2.3), (x+14, y-2.3), color=color, width=2.0)
    page.insert_text((x+18, y), label, fontsize=7.6,
                     fontname="legend", color=(0.1, 0.1, 0.1))
page.insert_text((3, map_height + 48), "Gray lines: road network", fontsize=7.6,
                 fontname="legend", color=(0.25, 0.25, 0.25))
result.set_metadata({"title": "Benchmark network and directional services",
                     "subject": "Source map retained; legend enlarged"})
out = ROOT / "figures" / "network_topology.pdf"
result.subset_fonts()
result.save(out, garbage=4, deflate=True)
page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(out.with_suffix(".png"))
print(out)
