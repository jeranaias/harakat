#!/usr/bin/env python3
"""
Generate demo GIF for Harakat README
Creates an animated terminal-style GIF showing diacritization examples
with realistic typing animation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from harakat import diacritize


def shape_arabic(text):
    """Reshape Arabic text for proper connected letter display"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


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
CURSOR_COLOR = (192, 202, 245)  # Same as command


def get_fonts():
    """Load fonts for rendering"""
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
            cmd_arabic_font = ImageFont.truetype(path, 14)
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


def draw_terminal_base(draw, mono_font):
    """Draw the terminal window chrome (title bar, controls, status bar)"""
    # Title bar background
    draw.rectangle([0, 0, WIDTH, 35], fill=(36, 40, 59))

    # Window controls
    draw.ellipse([15, 12, 27, 24], fill=(247, 118, 142))
    draw.ellipse([35, 12, 47, 24], fill=(224, 175, 104))
    draw.ellipse([55, 12, 67, 24], fill=(158, 206, 106))

    # Title
    draw.text((WIDTH // 2, 18), "Harakat - Arabic Diacritization",
              font=mono_font, fill=HEADER_COLOR, anchor="mm")

    # Status bar
    draw.rectangle([0, HEIGHT - 30, WIDTH, HEIGHT], fill=(36, 40, 59))
    stats = "2.29% DER  |  99.997% Quran  |  6.7 MB  |  62x smaller than SOTA"
    draw.text((WIDTH // 2, HEIGHT - 15), stats, font=mono_font, fill=STAT_COLOR, anchor="mm")


def create_typing_frame(completed_examples, current_cmd_text, show_cursor,
                        mono_font, arabic_font, cmd_arabic_font):
    """Create a frame showing typing progress"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw_terminal_base(draw, mono_font)

    y = 55

    # Draw completed examples
    for (desc, arabic, output) in completed_examples:
        # Command line
        draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
        cmd_prefix = 'python harakat.py "'
        draw.text((35, y), cmd_prefix, font=mono_font, fill=COMMAND_COLOR)
        prefix_bbox = draw.textbbox((0, 0), cmd_prefix, font=mono_font)
        prefix_width = prefix_bbox[2] - prefix_bbox[0]
        shaped_arabic = shape_arabic(arabic)
        draw.text((35 + prefix_width, y), shaped_arabic, font=cmd_arabic_font, fill=COMMAND_COLOR)
        arabic_bbox = draw.textbbox((0, 0), shaped_arabic, font=cmd_arabic_font)
        arabic_width = arabic_bbox[2] - arabic_bbox[0]
        draw.text((35 + prefix_width + arabic_width, y), '"', font=mono_font, fill=COMMAND_COLOR)
        y += 25

        # Input label and text
        draw.text((20, y), "Input:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(arabic), font=arabic_font, fill=ARABIC_COLOR, anchor="ra")
        y += 30

        # Output label and text
        draw.text((20, y), "Output:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(output), font=arabic_font, fill=OUTPUT_COLOR, anchor="ra")
        y += 35

    # Draw current typing line if there's text
    if current_cmd_text is not None:
        draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
        draw.text((35, y), current_cmd_text, font=mono_font, fill=COMMAND_COLOR)

        # Draw cursor
        if show_cursor:
            text_bbox = draw.textbbox((0, 0), current_cmd_text, font=mono_font)
            cursor_x = 35 + (text_bbox[2] - text_bbox[0])
            draw.text((cursor_x, y), "█", font=mono_font, fill=CURSOR_COLOR)

    return img


def create_typing_arabic_frame(completed_examples, cmd_prefix, arabic_so_far, full_arabic,
                               show_cursor, mono_font, arabic_font, cmd_arabic_font):
    """Create a frame showing Arabic text being typed"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw_terminal_base(draw, mono_font)

    y = 55

    # Draw completed examples
    for (desc, arabic, output) in completed_examples:
        draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
        prefix = 'python harakat.py "'
        draw.text((35, y), prefix, font=mono_font, fill=COMMAND_COLOR)
        prefix_bbox = draw.textbbox((0, 0), prefix, font=mono_font)
        prefix_width = prefix_bbox[2] - prefix_bbox[0]
        shaped_arabic = shape_arabic(arabic)
        draw.text((35 + prefix_width, y), shaped_arabic, font=cmd_arabic_font, fill=COMMAND_COLOR)
        arabic_bbox = draw.textbbox((0, 0), shaped_arabic, font=cmd_arabic_font)
        arabic_width = arabic_bbox[2] - arabic_bbox[0]
        draw.text((35 + prefix_width + arabic_width, y), '"', font=mono_font, fill=COMMAND_COLOR)
        y += 25

        draw.text((20, y), "Input:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(arabic), font=arabic_font, fill=ARABIC_COLOR, anchor="ra")
        y += 30

        draw.text((20, y), "Output:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(output), font=arabic_font, fill=OUTPUT_COLOR, anchor="ra")
        y += 35

    # Draw current line with Arabic being typed
    draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
    draw.text((35, y), cmd_prefix, font=mono_font, fill=COMMAND_COLOR)

    prefix_bbox = draw.textbbox((0, 0), cmd_prefix, font=mono_font)
    prefix_width = prefix_bbox[2] - prefix_bbox[0]

    if arabic_so_far:
        shaped = shape_arabic(arabic_so_far)
        draw.text((35 + prefix_width, y), shaped, font=cmd_arabic_font, fill=COMMAND_COLOR)
        arabic_bbox = draw.textbbox((0, 0), shaped, font=cmd_arabic_font)
        cursor_x = 35 + prefix_width + (arabic_bbox[2] - arabic_bbox[0])
    else:
        cursor_x = 35 + prefix_width

    if show_cursor:
        draw.text((cursor_x, y), "█", font=mono_font, fill=CURSOR_COLOR)

    return img


def create_output_frame(completed_examples, current_arabic, current_output, show_output,
                        mono_font, arabic_font, cmd_arabic_font):
    """Create a frame showing the output appearing"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw_terminal_base(draw, mono_font)

    y = 55

    # Draw completed examples
    for (desc, arabic, output) in completed_examples:
        draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
        prefix = 'python harakat.py "'
        draw.text((35, y), prefix, font=mono_font, fill=COMMAND_COLOR)
        prefix_bbox = draw.textbbox((0, 0), prefix, font=mono_font)
        prefix_width = prefix_bbox[2] - prefix_bbox[0]
        shaped_arabic = shape_arabic(arabic)
        draw.text((35 + prefix_width, y), shaped_arabic, font=cmd_arabic_font, fill=COMMAND_COLOR)
        arabic_bbox = draw.textbbox((0, 0), shaped_arabic, font=cmd_arabic_font)
        arabic_width = arabic_bbox[2] - arabic_bbox[0]
        draw.text((35 + prefix_width + arabic_width, y), '"', font=mono_font, fill=COMMAND_COLOR)
        y += 25

        draw.text((20, y), "Input:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(arabic), font=arabic_font, fill=ARABIC_COLOR, anchor="ra")
        y += 30

        draw.text((20, y), "Output:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(output), font=arabic_font, fill=OUTPUT_COLOR, anchor="ra")
        y += 35

    # Draw current command (complete)
    draw.text((20, y), "$", font=mono_font, fill=PROMPT_COLOR)
    prefix = 'python harakat.py "'
    draw.text((35, y), prefix, font=mono_font, fill=COMMAND_COLOR)
    prefix_bbox = draw.textbbox((0, 0), prefix, font=mono_font)
    prefix_width = prefix_bbox[2] - prefix_bbox[0]
    shaped_arabic = shape_arabic(current_arabic)
    draw.text((35 + prefix_width, y), shaped_arabic, font=cmd_arabic_font, fill=COMMAND_COLOR)
    arabic_bbox = draw.textbbox((0, 0), shaped_arabic, font=cmd_arabic_font)
    arabic_width = arabic_bbox[2] - arabic_bbox[0]
    draw.text((35 + prefix_width + arabic_width, y), '"', font=mono_font, fill=COMMAND_COLOR)
    y += 25

    if show_output:
        # Input
        draw.text((20, y), "Input:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(current_arabic), font=arabic_font, fill=ARABIC_COLOR, anchor="ra")
        y += 30

        # Output
        draw.text((20, y), "Output:", font=mono_font, fill=LABEL_COLOR)
        draw.text((WIDTH - 30, y), shape_arabic(current_output), font=arabic_font, fill=OUTPUT_COLOR, anchor="ra")

    return img


def main():
    print("Generating demo GIF with typing animation...")

    mono_font, arabic_font, cmd_arabic_font = get_fonts()

    # Examples to show
    examples = [
        ("Basmala", "بسم الله الرحمن الرحيم", diacritize("بسم الله الرحمن الرحيم")),
        ("Proverb", "من جد وجد ومن زرع حصد", diacritize("من جد وجد ومن زرع حصد")),
        ("Education", "ذهب الطالب الى المدرسة", diacritize("ذهب الطالب الى المدرسة")),
    ]

    frames = []
    durations = []

    cmd_prefix = 'python harakat.py "'

    # Initial frame - empty terminal with blinking cursor
    for i in range(2):
        frames.append(create_typing_frame([], "", i % 2 == 0, mono_font, arabic_font, cmd_arabic_font))
        durations.append(300)

    completed = []

    for example_idx, (desc, arabic, output) in enumerate(examples):
        # Type out the command prefix character by character
        for i in range(1, len(cmd_prefix) + 1):
            frames.append(create_typing_frame(completed, cmd_prefix[:i], True,
                                             mono_font, arabic_font, cmd_arabic_font))
            durations.append(50)  # Fast typing

        # Type out the Arabic text character by character
        for i in range(1, len(arabic) + 1):
            frames.append(create_typing_arabic_frame(completed, cmd_prefix, arabic[:i], arabic, True,
                                                     mono_font, arabic_font, cmd_arabic_font))
            durations.append(60)  # Slightly slower for Arabic

        # Type the closing quote
        frames.append(create_typing_frame(completed, cmd_prefix + shape_arabic(arabic) + '"', True,
                                         mono_font, arabic_font, cmd_arabic_font))
        durations.append(100)

        # Cursor blink before Enter
        for i in range(2):
            frames.append(create_output_frame(completed, arabic, output, False,
                                             mono_font, arabic_font, cmd_arabic_font))
            durations.append(200)

        # Show output (instant appear)
        frames.append(create_output_frame(completed, arabic, output, True,
                                         mono_font, arabic_font, cmd_arabic_font))
        durations.append(1200)  # Pause to read

        # Add to completed
        completed.append((desc, arabic, output))

    # Final pause on completed state
    frames.append(create_output_frame(completed[:-1], examples[-1][1], examples[-1][2], True,
                                     mono_font, arabic_font, cmd_arabic_font))
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
    print(f"Frames: {len(frames)}")


if __name__ == "__main__":
    main()
