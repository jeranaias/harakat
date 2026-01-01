# Harakat Examples

This directory contains example files demonstrating Harakat's capabilities.

## Files

- **harakat_demo.ipynb** - Interactive Google Colab notebook
- **demo.py** - Terminal demo script with colorful output
- **demo.tape** - VHS tape file for recording GIF demos

## Recording a Terminal Demo GIF

### Option 1: VHS (Recommended)

[VHS](https://github.com/charmbracelet/vhs) creates beautiful terminal GIFs from tape files.

```bash
# Install VHS (requires Go)
go install github.com/charmbracelet/vhs@latest

# Record the demo
cd examples
vhs demo.tape
```

This generates `demo.gif` ready for the README.

### Option 2: asciinema + svg-term

```bash
# Install asciinema
pip install asciinema

# Record the demo
asciinema rec demo.cast

# Convert to SVG/GIF
npx svg-term --in demo.cast --out demo.svg
```

### Option 3: termtosvg

```bash
# Install termtosvg
pip install termtosvg

# Record
termtosvg demo.svg -c "python demo.py"
```

### Option 4: Screen Recording

Use OBS, ScreenToGif, or similar tools to record the terminal while running:

```bash
python examples/demo.py
```

## Running the Demo Script

```bash
cd harakat
python examples/demo.py
```

The script demonstrates:
- Quranic text diacritization (99.997% accuracy)
- Modern Standard Arabic (2.29% DER)
- Arabic proverbs
- Performance benchmarks
