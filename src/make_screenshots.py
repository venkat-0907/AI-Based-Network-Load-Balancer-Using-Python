"""
make_screenshots.py
--------------------
Renders the REAL captured console output (from outputs/logs/*.txt) into
terminal-styled PNG images for the README "Screenshots" section. This is
not fake data -- it's the actual stdout produced by running train_model.py,
predict.py, and load_balancer_simulator.py, just styled to look like a
terminal window screenshot.
"""

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def get_font(size=15, bold=False):
    path = FONT_CANDIDATES[1] if bold else FONT_CANDIDATES[0]
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_terminal(text_path, out_path, title="terminal", max_lines=60, width=1150):
    with open(text_path) as f:
        lines = f.read().splitlines()
    lines = [ln for ln in lines if not ln.strip().startswith("Passing `palette`")
              and "FutureWarning" not in ln and ln.strip() != ""]
    lines = lines[:max_lines]

    font = get_font(14)
    title_font = get_font(13, bold=True)
    line_height = 20
    top_bar_height = 34
    padding = 18

    height = top_bar_height + padding * 2 + line_height * len(lines)
    img = Image.new("RGB", (width, height), color=(30, 30, 32))
    draw = ImageDraw.Draw(img)

    # top bar
    draw.rectangle([0, 0, width, top_bar_height], fill=(44, 44, 48))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([16 + i * 22, 11, 16 + i * 22 + 12, 23], fill=c)
    draw.text((width / 2 - len(title) * 3.6, 9), title, font=title_font, fill=(220, 220, 220))

    y = top_bar_height + padding
    for line in lines:
        color = (198, 246, 213) if (">>>" in line or "ROUTE NEXT" in line) else (223, 223, 223)
        if line.strip().startswith("=") :
            color = (120, 190, 255)
        draw.text((padding, y), line, font=font, fill=color)
        y += line_height

    img.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    render_terminal("outputs/logs/train_console_output.txt",
                     "outputs/screenshots/01_training_console.png",
                     title="python3 src/train_model.py", max_lines=45)

    render_terminal("outputs/logs/predict_demo_output.txt",
                     "outputs/screenshots/02_prediction_console.png",
                     title="python3 src/predict.py", max_lines=30)

    render_terminal("outputs/logs/simulator_console_output.txt",
                     "outputs/screenshots/03_simulator_console.png",
                     title="python3 src/load_balancer_simulator.py", max_lines=30)
