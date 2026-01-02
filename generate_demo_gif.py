#!/usr/bin/env python3
"""
Generate demo GIF for Harakat README
Creates an animated terminal-style GIF showing diacritization examples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from harakat import diacritize

# Configuration
WIDTH = 800
HEIGHT = 450
BG_COLOR = (26, 27, 38)  # Tokyo Night background
PROMPT_COLOR = (158, 206, 106)  # Green
COMMAND_COLOR = (192, 202, 245)  # Light blue
LABEL_COLOR = (187, 154, 247)  # Purple
ARABIC_COLOR = (125, 207, 255)  # Cyan
OUTPUT_COLOR = (158, 206, 106)  # Green
STAT_COLOR = (86, 95, 137)  # Gray
HEADER_COLOR = (122, 162, 247)  # Blue

# Try to load fonts
def get_fonts():
    # Try common font paths
    mono_paths = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    arabic_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    mono_font = None
    arabic_font = None
    cmd_arabic_font = None

    for path in mono_paths:
        try:
            mono_font = ImageFont.truetype(path, 14)
            break
        except:
            pass

    for path in arabic_paths:
        try:
            arabic_font = ImageFont.truetype(path, 20)
            cmd_arabic_font = ImageFont.truetype(path, 14)  # Smaller for command line
            break
        except:
            pass

    if mono_font is None:
        mono_font = ImageFont.load_default()
    if arabic_font is None:
        arabic_font = ImageFont.load_default()
    if cmd_arabic_font is None:
        cmd_arabic_font = ImageFont.load_default()

    return mono_font, arabic_font, cmd_arabic_font

def create_frame(examples, current_example, show_output, mono_font, arabic_font, cmd_arabic_font):
    """Create a single frame of the animation"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title bar background
    draw.rectangle([0, 0, WIDTH, 35], fill=(36, 40, 59))

    # Window controls
    draw.ellipse([15, 12, 27, 24], fill=(247, 118, 142))
    draw.ellipse([35, 12, 47, 24], fill=(224, 175, 104))
    draw.ellipse([55, 12, 67, 24], fill=(158, 206, 106))

    # Title
    draw.text((WIDTH // 2, 18), "Harakat - Arabic Diacritization",
              font=mono_font, fill=HEADER_COLOR, anchor="mm")

    y = 55

    for i, (desc, arabic, output) in enumerate(examples):
        if i > current_example:
            break

        # Prompt and command - split into parts to use proper fonts
        draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
        cmd_prefix = 'python harakat.py "'
        draw.text((35, y), cmd_prefix, font=mono_font, fill=COMMAND_COLOR)
        # Calculate width of prefix to position Arabic text
        prefix_bbox = draw.textbbox((0, 0), cmd_prefix, font=mono_font)
        prefix_width = prefix_bbox[2] - prefix_bbox[0]
        # Draw Arabic with proper font
        draw.text((35 + prefix_width, y), arabic, font=cmd_arabic_font, fill=COMMAND_COLOR)
        # Calculate Arabic width and add closing quote
        arabic_bbox = draw.textbbox((0, 0), arabic, font=cmd_arabic_font)
        arabic_width = arabic_bbox[2] - arabic_bbox[0]
        draw.text((35 + prefix_width + arabic_width, y), '"', font=mono_font, fill=COMMAND_COLOR)
        y += 25

        if i < current_example or (i == current_example and show_output):
            # Input label
            draw.text((20, y), "Input:", font=mono_font, fill=LABEL_COLOR)
            # Arabic text (right-to-left)
            draw.text((WIDTH - 30, y), arabic, font=arabic_font, fill=ARABIC_COLOR, anchor="ra")
            y += 30

            # Output label
            draw.text((20, y), "Output:", font=mono_font, fill=LABEL_COLOR)
            # Diacritized text
            draw.text((WIDTH - 30, y), output, font=arabic_font, fill=OUTPUT_COLOR, anchor="ra")
            y += 35
        else:
            y += 65

    # Status bar
    draw.rectangle([0, HEIGHT - 30, WIDTH, HEIGHT], fill=(36, 40, 59))
    stats = "2.29% DER  |  99.997% Quran  |  6.7 MB  |  62x smaller than SOTA  |  github.com/jeranaias/harakat"
    draw.text((WIDTH // 2, HEIGHT - 15), stats, font=mono_font, fill=STAT_COLOR, anchor="mm")

    return img

def main():
    print("Generating demo GIF...")

    mono_font, arabic_font, cmd_arabic_font = get_fonts()

    # Examples to show
    examples = [
        ("Basmala", "بسم الله الرحمن الرحيم", diacritize("بسم الله الرحمن الرحيم")),
        ("Proverb", "من جد وجد ومن زرع حصد", diacritize("من جد وجد ومن زرع حصد")),
        ("Education", "ذهب الطالب الى المدرسة", diacritize("ذهب الطالب الى المدرسة")),
    ]

    frames = []
    durations = []

    # Initial blank frame
    frames.append(create_frame(examples, -1, False, mono_font, arabic_font, cmd_arabic_font))
    durations.append(500)

    # Animate each example
    for i in range(len(examples)):
        # Show command
        frames.append(create_frame(examples, i, False, mono_font, arabic_font, cmd_arabic_font))
        durations.append(800)

        # Show output
        frames.append(create_frame(examples, i, True, mono_font, arabic_font, cmd_arabic_font))
        durations.append(1500)

    # Final pause
    frames.append(create_frame(examples, len(examples) - 1, True, mono_font, arabic_font, cmd_arabic_font))
    durations.append(2000)

    # Save as GIF
    output_path = os.path.join(os.path.dirname(__file__), 'docs', 'demo.gif')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )

    print(f"Saved to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
