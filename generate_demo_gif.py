#!/usr/bin/env python3
"""
Generate demo GIF for Harakat README
Creates an animated terminal-style GIF showing Python REPL workflow
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
WIDTH = 900
HEIGHT = 550
BG_COLOR = (26, 27, 38)  # Tokyo Night background
PROMPT_COLOR = (158, 206, 106)  # Green for $ and >>>
COMMAND_COLOR = (192, 202, 245)  # Light blue for commands
KEYWORD_COLOR = (187, 154, 247)  # Purple for keywords
STRING_COLOR = (158, 206, 106)  # Green for strings
FUNCTION_COLOR = (224, 175, 104)  # Orange/yellow for functions
OUTPUT_COLOR = (125, 207, 255)  # Cyan for output
COMMENT_COLOR = (86, 95, 137)  # Gray for comments
HEADER_COLOR = (122, 162, 247)  # Blue
CURSOR_COLOR = (192, 202, 245)  # Cursor


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

    for path in mono_paths:
        try:
            mono_font = ImageFont.truetype(path, 14)
            break
        except:
            pass

    for path in arabic_paths:
        try:
            arabic_font = ImageFont.truetype(path, 16)
            break
        except:
            pass

    if mono_font is None:
        mono_font = ImageFont.load_default()
    if arabic_font is None:
        arabic_font = ImageFont.load_default()

    return mono_font, arabic_font


def draw_terminal_chrome(draw, mono_font):
    """Draw the terminal window chrome"""
    # Title bar background
    draw.rectangle([0, 0, WIDTH, 35], fill=(36, 40, 59))

    # Window controls
    draw.ellipse([15, 12, 27, 24], fill=(247, 118, 142))
    draw.ellipse([35, 12, 47, 24], fill=(224, 175, 104))
    draw.ellipse([55, 12, 67, 24], fill=(158, 206, 106))

    # Title
    draw.text((WIDTH // 2, 18), "Python 3.12 - Harakat Demo",
              font=mono_font, fill=HEADER_COLOR, anchor="mm")


class TerminalRenderer:
    """Renders terminal content with syntax highlighting"""

    def __init__(self, mono_font, arabic_font):
        self.mono_font = mono_font
        self.arabic_font = arabic_font
        self.line_height = 22
        self.start_y = 50
        self.start_x = 20

    def create_frame(self, lines, cursor_line=None, cursor_pos=None, show_cursor=True):
        """Create a frame with the given lines"""
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        draw_terminal_chrome(draw, self.mono_font)

        y = self.start_y

        for line_idx, line in enumerate(lines):
            x = self.start_x
            x = self.render_line(draw, line, x, y)

            # Draw cursor if this is the cursor line
            if show_cursor and cursor_line == line_idx and cursor_pos is not None:
                cursor_x = self.start_x + self.get_text_width(draw, line['text'][:cursor_pos])
                draw.text((cursor_x, y), "█", font=self.mono_font, fill=CURSOR_COLOR)

            y += self.line_height

        # Draw cursor at end of last line if specified
        if show_cursor and cursor_line == len(lines) and cursor_pos == 0:
            draw.text((self.start_x, y), "█", font=self.mono_font, fill=CURSOR_COLOR)

        return img

    def render_line(self, draw, line, x, y):
        """Render a single line with syntax highlighting"""
        line_type = line.get('type', 'text')
        text = line.get('text', '')

        if line_type == 'shell':
            # Shell command: $ python
            draw.text((x, y), "$ ", font=self.mono_font, fill=PROMPT_COLOR)
            x += self.get_text_width(draw, "$ ")
            draw.text((x, y), text, font=self.mono_font, fill=COMMAND_COLOR)

        elif line_type == 'repl_start':
            # Python REPL startup message
            draw.text((x, y), text, font=self.mono_font, fill=COMMENT_COLOR)

        elif line_type == 'prompt':
            # >>> prompt with code
            draw.text((x, y), ">>> ", font=self.mono_font, fill=PROMPT_COLOR)
            x += self.get_text_width(draw, ">>> ")
            x = self.render_python_code(draw, text, x, y)

        elif line_type == 'continuation':
            # ... continuation prompt
            draw.text((x, y), "... ", font=self.mono_font, fill=PROMPT_COLOR)
            x += self.get_text_width(draw, "... ")
            draw.text((x, y), text, font=self.mono_font, fill=COMMAND_COLOR)

        elif line_type == 'output':
            # Output text (may contain Arabic)
            self.render_output(draw, text, x, y)

        elif line_type == 'output_arabic':
            # Arabic output - render right-to-left
            shaped = shape_arabic(text)
            draw.text((x, y), shaped, font=self.arabic_font, fill=OUTPUT_COLOR)

        elif line_type == 'comment':
            # Comment line
            draw.text((x, y), text, font=self.mono_font, fill=COMMENT_COLOR)

        elif line_type == 'blank':
            pass  # Empty line

        return x

    def render_python_code(self, draw, code, x, y):
        """Render Python code with syntax highlighting"""
        keywords = ['from', 'import', 'print', 'for', 'in', 'if', 'else', 'def', 'return', 'True', 'False', 'None']
        functions = ['diacritize', 'len', 'print', 'range']

        i = 0
        while i < len(code):
            # Check for string
            if code[i] in '"\'':
                quote = code[i]
                end = code.find(quote, i + 1)
                if end == -1:
                    end = len(code) - 1
                string_content = code[i:end + 1]

                # Check if string contains Arabic
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in string_content)

                if has_arabic:
                    # Render quote, then Arabic, then quote
                    draw.text((x, y), quote, font=self.mono_font, fill=STRING_COLOR)
                    x += self.get_text_width(draw, quote)

                    arabic_part = string_content[1:-1]
                    shaped = shape_arabic(arabic_part)
                    draw.text((x, y), shaped, font=self.arabic_font, fill=STRING_COLOR)
                    x += self.get_text_width(draw, shaped, self.arabic_font)

                    draw.text((x, y), quote, font=self.mono_font, fill=STRING_COLOR)
                    x += self.get_text_width(draw, quote)
                else:
                    draw.text((x, y), string_content, font=self.mono_font, fill=STRING_COLOR)
                    x += self.get_text_width(draw, string_content)

                i = end + 1
                continue

            # Check for word (keyword, function, or identifier)
            if code[i].isalpha() or code[i] == '_':
                word_end = i
                while word_end < len(code) and (code[word_end].isalnum() or code[word_end] == '_'):
                    word_end += 1
                word = code[i:word_end]

                if word in keywords:
                    color = KEYWORD_COLOR
                elif word in functions:
                    color = FUNCTION_COLOR
                else:
                    color = COMMAND_COLOR

                draw.text((x, y), word, font=self.mono_font, fill=color)
                x += self.get_text_width(draw, word)
                i = word_end
                continue

            # Regular character
            draw.text((x, y), code[i], font=self.mono_font, fill=COMMAND_COLOR)
            x += self.get_text_width(draw, code[i])
            i += 1

        return x

    def render_output(self, draw, text, x, y):
        """Render output which may contain Arabic"""
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)

        if has_arabic:
            shaped = shape_arabic(text)
            draw.text((x, y), shaped, font=self.arabic_font, fill=OUTPUT_COLOR)
        else:
            draw.text((x, y), text, font=self.mono_font, fill=OUTPUT_COLOR)

    def get_text_width(self, draw, text, font=None):
        """Get the width of text"""
        if font is None:
            font = self.mono_font
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def type_text(frames, durations, renderer, base_lines, new_line, char_delay=40):
    """Animate typing a new line character by character"""
    line_type = new_line.get('type', 'text')
    full_text = new_line.get('text', '')

    # Determine the prefix based on line type
    if line_type == 'prompt':
        prefix = ">>> "
    elif line_type == 'shell':
        prefix = "$ "
    elif line_type == 'continuation':
        prefix = "... "
    else:
        prefix = ""

    # Type character by character
    for i in range(len(full_text) + 1):
        partial_line = {**new_line, 'text': full_text[:i]}
        current_lines = base_lines + [partial_line]

        frame = renderer.create_frame(current_lines, cursor_line=len(base_lines),
                                       cursor_pos=len(prefix) + i, show_cursor=True)
        frames.append(frame)
        durations.append(char_delay)


def add_static_frame(frames, durations, renderer, lines, duration, show_cursor=False):
    """Add a static frame"""
    frame = renderer.create_frame(lines, show_cursor=show_cursor)
    frames.append(frame)
    durations.append(duration)


def main():
    print("Generating developer workflow GIF...")

    mono_font, arabic_font = get_fonts()
    renderer = TerminalRenderer(mono_font, arabic_font)

    frames = []
    durations = []

    # Build up the terminal session step by step
    lines = []

    # === Scene 1: Start Python REPL ===
    print("  Scene 1: Starting Python REPL...")

    # Show empty terminal with cursor
    add_static_frame(frames, durations, renderer, [], 500, show_cursor=True)

    # Type: $ python
    type_text(frames, durations, renderer, [], {'type': 'shell', 'text': 'python'}, char_delay=60)
    add_static_frame(frames, durations, renderer, [{'type': 'shell', 'text': 'python'}], 300)

    # Show Python startup
    lines = [
        {'type': 'shell', 'text': 'python'},
        {'type': 'repl_start', 'text': 'Python 3.12.0 (main) [GCC 11.4.0]'},
        {'type': 'repl_start', 'text': 'Type "help" for more information.'},
    ]
    add_static_frame(frames, durations, renderer, lines, 600)

    # === Scene 2: Import harakat ===
    print("  Scene 2: Importing harakat...")

    # Type import statement
    import_line = {'type': 'prompt', 'text': 'from harakat import diacritize'}
    type_text(frames, durations, renderer, lines, import_line, char_delay=35)
    lines.append(import_line)
    add_static_frame(frames, durations, renderer, lines, 400)

    # === Scene 3: First example - Basic usage ===
    print("  Scene 3: Basic diacritization example...")

    # Type first diacritize call
    example1 = {'type': 'prompt', 'text': 'diacritize("ذهب الطالب الى المدرسة")'}
    type_text(frames, durations, renderer, lines, example1, char_delay=40)
    lines.append(example1)
    add_static_frame(frames, durations, renderer, lines, 300)

    # Show output
    output1 = diacritize("ذهب الطالب الى المدرسة")
    lines.append({'type': 'output_arabic', 'text': f"'{output1}'"})
    add_static_frame(frames, durations, renderer, lines, 1200)

    # === Scene 4: Quran example ===
    print("  Scene 4: Quran example with auto-detection...")

    # Add comment
    lines.append({'type': 'blank', 'text': ''})
    comment1 = {'type': 'prompt', 'text': '# Quran text is auto-detected with 99.997% accuracy'}
    type_text(frames, durations, renderer, lines, comment1, char_delay=30)
    lines.append(comment1)
    add_static_frame(frames, durations, renderer, lines, 400)

    # Type Quran example
    example2 = {'type': 'prompt', 'text': 'diacritize("بسم الله الرحمن الرحيم")'}
    type_text(frames, durations, renderer, lines, example2, char_delay=40)
    lines.append(example2)
    add_static_frame(frames, durations, renderer, lines, 300)

    # Show output
    output2 = diacritize("بسم الله الرحمن الرحيم")
    lines.append({'type': 'output_arabic', 'text': f"'{output2}'"})
    add_static_frame(frames, durations, renderer, lines, 1200)

    # === Scene 5: Print example ===
    print("  Scene 5: Print statement example...")

    example3 = {'type': 'prompt', 'text': 'print(diacritize("العلم نور والجهل ظلام"))'}
    type_text(frames, durations, renderer, lines, example3, char_delay=35)
    lines.append(example3)
    add_static_frame(frames, durations, renderer, lines, 300)

    # Show printed output (no quotes)
    output3 = diacritize("العلم نور والجهل ظلام")
    lines.append({'type': 'output_arabic', 'text': output3})
    add_static_frame(frames, durations, renderer, lines, 1200)

    # === Scene 6: Batch processing ===
    print("  Scene 6: Batch processing example...")

    lines.append({'type': 'blank', 'text': ''})
    comment2 = {'type': 'prompt', 'text': '# Batch processing'}
    type_text(frames, durations, renderer, lines, comment2, char_delay=35)
    lines.append(comment2)
    add_static_frame(frames, durations, renderer, lines, 300)

    batch_line = {'type': 'prompt', 'text': 'texts = ["مرحبا", "كيف حالك", "شكرا جزيلا"]'}
    type_text(frames, durations, renderer, lines, batch_line, char_delay=35)
    lines.append(batch_line)
    add_static_frame(frames, durations, renderer, lines, 300)

    batch_process = {'type': 'prompt', 'text': '[diacritize(t) for t in texts]'}
    type_text(frames, durations, renderer, lines, batch_process, char_delay=40)
    lines.append(batch_process)
    add_static_frame(frames, durations, renderer, lines, 300)

    # Show batch output
    batch_outputs = [diacritize(t) for t in ["مرحبا", "كيف حالك", "شكرا جزيلا"]]
    batch_result = "['" + "', '".join(batch_outputs) + "']"
    lines.append({'type': 'output_arabic', 'text': batch_result})
    add_static_frame(frames, durations, renderer, lines, 1500)

    # === Scene 7: Famous proverb ===
    print("  Scene 7: Famous proverb...")

    lines.append({'type': 'blank', 'text': ''})
    example4 = {'type': 'prompt', 'text': 'diacritize("من جد وجد ومن زرع حصد")'}
    type_text(frames, durations, renderer, lines, example4, char_delay=40)
    lines.append(example4)
    add_static_frame(frames, durations, renderer, lines, 300)

    output4 = diacritize("من جد وجد ومن زرع حصد")
    lines.append({'type': 'output_arabic', 'text': f"'{output4}'"})
    add_static_frame(frames, durations, renderer, lines, 1500)

    # === Final pause ===
    add_static_frame(frames, durations, renderer, lines, 2500)

    # Save as GIF
    output_path = os.path.join(os.path.dirname(__file__), 'docs', 'demo.gif')

    print(f"  Saving {len(frames)} frames...")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

    file_size = os.path.getsize(output_path)
    print(f"\nSaved to {output_path}")
    print(f"Size: {file_size / 1024:.1f} KB")
    print(f"Frames: {len(frames)}")

    # Calculate total duration
    total_ms = sum(durations)
    print(f"Duration: {total_ms / 1000:.1f} seconds")


if __name__ == "__main__":
    main()
