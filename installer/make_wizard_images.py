"""Regenerate the Inno Setup wizard bitmaps from gui/assets.

Run after changing the logo: python installer/make_wizard_images.py

Inno picks one file per set by matching the display's DPI scaling, so each set
ships at several sizes. BMP rather than PNG: PNG only works on Inno 6.3+, and
BMP is understood by every version.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "gui" / "assets"
OUT = Path(__file__).resolve().parent

# The splash card colour from gui/app.py, so the installer and the app that
# follows it share a background.
DARK = (30, 34, 40)
# The small image sits in the wizard's page header, which is white under
# WizardStyle=modern.
LIGHT = (255, 255, 255)

# Inno's standard scaled sizes for each slot.
LARGE = [(164, 314), (246, 471), (328, 628)]
SMALL = [(55, 55), (110, 110), (164, 164)]


def compose(logo: Image.Image, size: tuple[int, int], bg, margin: float):
    """Fit the logo inside `size` on a flat background, preserving aspect."""
    width, height = size
    box_w = int(width * (1 - 2 * margin))
    scaled_h = round(box_w * logo.height / logo.width)
    # A tall logo in a wide slot has to be bounded on height instead.
    box_h = int(height * (1 - 2 * margin))
    if scaled_h > box_h:
        box_w = round(box_h * logo.width / logo.height)
        scaled_h = box_h

    canvas = Image.new("RGB", size, bg)
    scaled = logo.resize((box_w, scaled_h), Image.LANCZOS)
    # Composited through its own alpha so the transparent border does not
    # darken the background.
    canvas.paste(scaled, ((width - box_w) // 2, (height - scaled_h) // 2), scaled)
    return canvas


def main() -> None:
    splash = Image.open(ASSETS / "splash.png").convert("RGBA")
    icon = Image.open(ASSETS / "icon.png").convert("RGBA")
    icon = icon.crop(icon.getbbox())

    for size in LARGE:
        compose(splash, size, DARK, 0.10).save(OUT / f"wizard-large-{size[0]}.bmp")
    for size in SMALL:
        compose(icon, size, LIGHT, 0.08).save(OUT / f"wizard-small-{size[0]}.bmp")

    for path in sorted(OUT.glob("wizard-*.bmp")):
        print(f"{path.name:24} {path.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
