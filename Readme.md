# AI Meme Generator

A **Flask**-based web application that generates funny memes from user-provided images. The user can provide both the image and captions or it also has an option to create ai captions.Ai-Captions are generated in two steps:

1. **BLIP** (Salesforce) produces an image description.
2. **Flan-T5-Large** refines the description into a witty, two-line meme caption (setup + punchline).

All processing runs on the CPU under 16 GB of RAM—no GPU required.

---

##  Directory Structure

```
├── app.py               # Main Flask application
├── requirements.txt     # List of Python dependencies
├── README.md            # This file
├── templates/
│   ├── index.html       # Upload form & AI-suggest button
│   └── result.html      # Meme display & download links
└── static/
    ├── css/
    │   └── style.css    # Styles for form, buttons, layout
    ├── fonts/
    │   └── Impact.ttf   # Impact font for meme text
    └── memes/           # Generated memes (output images)
```

---

## Getting Started

### Prerequisites

* Python 3.8+
* 16 GB RAM (to load Flan-T5-Large in CPU mode)

### Installation

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>
   cd meme-ai
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # on Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

1. **Start the Flask server**

   ```bash
   python app.py
   ```

2. **Open** your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Usage**

   * Upload an image (PNG/JPG/GIF).
   * (Optional) Click **Suggest AI Caption** to auto-fill top/bottom text.
   * Click **Generate Meme** to overlay text and view/download your meme.

 Contributing

Feel free to open issues or pull requests to improve the meme captions, UI/UX, or performance.

---

*Enjoy creating hilarious memes!*
