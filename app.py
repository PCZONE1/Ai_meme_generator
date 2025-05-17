import os
import torch
print("1️⃣  Starting import…")

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# ─── Config ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MEME_FOLDER']   = 'static/memes'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MEME_FOLDER'], exist_ok=True)

# ─── Load AI Models Locally ─────────────────────────────────────────
print("2️⃣  Loading BLIP image-captioning model…")
processor  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("3️⃣  Initializing Flan-T5-Large for meme captions…")
instruct = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device_map={"": "cpu"},
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)
print("✅ Models loaded; Flask starting now.")

# ─── Helpers ────────────────────────────────────────────────────────
def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def ai_suggest_caption(image_path):
    """Generate a descriptive caption for the image using BLIP."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)


def hf_caption(raw_caption):
    """Transform raw caption into a witty two-line meme caption."""
    prompt = (
        "You are a professional meme writer. "
        f"The image description is: '{raw_caption}'.\n"
        "Write a funny two-line meme caption in conversational, internet meme style: "
        "the first line should set up a relatable scenario, and the second line should deliver a punchline."
    )
    res = instruct(prompt)
    text = res[0]['generated_text'].strip().split("\n")
    top = text[0].upper() if len(text) > 0 else ''
    bottom = text[1].upper() if len(text) > 1 else ''
    return top, bottom


def wrap_text(text, draw, font, max_width):
    words = text.split()
    if not words:
        return []
    lines = []
    line = words[0]
    for word in words[1:]:
        test_line = f"{line} {word}"
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def draw_wrapped_text(draw, img, lines, font, y_start):
    outline = max(int(font.size / 15), 1)
    y = y_start
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (img.width - w) // 2
        # outline
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                draw.text((x + dx, y + dy), line, font=font, fill='black')
        draw.text((x, y), line, font=font, fill='white')
        y += h + 5
    return y - y_start


# ─── Routes ─────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/ai-captions', methods=['POST'])
def ai_caps():
    img = request.files.get('image')
    if not img or not allowed_file(img.filename):
        return jsonify(error="Invalid image"), 400

    fname = secure_filename(img.filename)
    tmp = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    img.save(tmp)

    raw = ai_suggest_caption(tmp)
    top, bottom = hf_caption(raw)
    return jsonify(top=top, bottom=bottom)


@app.route('/generate', methods=['POST'])
def generate():
    img_file = request.files.get('image')
    top_text = request.form.get('top_text', '').upper()
    bottom_text = request.form.get('bottom_text', '').upper()

    if not img_file or not allowed_file(img_file.filename):
        return redirect(url_for('index'))

    fname = secure_filename(img_file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    img_file.save(in_path)

    if not top_text and not bottom_text:
        raw = ai_suggest_caption(in_path)
        top_text, bottom_text = hf_caption(raw)

    img = Image.open(in_path)
    draw = ImageDraw.Draw(img)
    font_size = max(int(img.width / 12), 24)
    font = ImageFont.truetype('static/fonts/Impact.ttf', font_size)
    max_text_width = img.width - 20

    top_lines = wrap_text(top_text, draw, font, max_text_width)
    draw_wrapped_text(draw, img, top_lines, font, y_start=10)

    bottom_lines = wrap_text(bottom_text, draw, font, max_text_width)
    block_h = sum(
        (draw.textbbox((0,0), l, font=font)[3] - draw.textbbox((0,0), l, font=font)[1] + 5)
        for l in bottom_lines) - 5
    y_start = img.height - block_h - 10
    draw_wrapped_text(draw, img, bottom_lines, font, y_start=y_start)

    meme_name = f"meme_{fname}"
    out_path = os.path.join(app.config['MEME_FOLDER'], meme_name)
    img.save(out_path)

    return render_template('result.html', meme_file=meme_name)


if __name__ == '__main__':
    app.run(debug=True)

