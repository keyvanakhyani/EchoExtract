# EchoExtract

**Transcribe Persian and English videos to text using local Whisper — with clean, layered architecture.**

EchoExtract turns your video files into text. It extracts the audio, runs it through a locally-hosted Whisper model (GPU-accelerated), and writes the transcript in multiple formats (plain text, SRT/VTT subtitles, and timestamped JSON). Language is auto-detected, so both Persian and English work out of the box.

---

## ✨ Features

- 🎬 **Video → text** in one command (mp4 and other formats via ffmpeg)
- 🌍 **Persian & English** — automatic language detection (powered by Whisper `large-v3`)
- 🖥️ **Runs locally on GPU** — no data leaves your machine, no API costs
- 📄 **Multiple output formats** — `txt`, `srt`, `vtt`, `json` (choose which you want)
- 🧩 **Pluggable engine architecture** — swap the transcription backend without touching the rest of the app
- 🧼 **Clean, typed, layered codebase** — Pydantic models, an abstract engine interface, and a thin CLI

---

## 🏗️ Architecture

The project is organized into independent layers, each with a single responsibility:

```
src/echo_extract/
├── core/
│   └── models.py          # Pydantic domain models (Segment, TranscriptionResult)
├── engines/
│   ├── base.py            # Abstract TranscriptionEngine interface (Strategy pattern)
│   └── faster_whisper_engine.py   # Local GPU engine (faster-whisper / CTranslate2)
├── io/
│   ├── audio.py           # Audio extraction from video (ffmpeg)
│   └── writers.py         # Output writers (txt / srt / vtt / json)
├── pipeline.py            # Orchestration: video → audio → transcript → files
└── cli.py                 # Command-line interface
```

**Why this design?** The `pipeline` holds the pure logic and knows nothing about *how* it's called. The CLI (and future UIs like Gradio or Telegram) simply call `transcribe_video()` — no code duplication. Engines implement a shared interface, so a cloud backend can be added later without changing the pipeline.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and on your PATH ([download](https://ffmpeg.org/download.html))
- **NVIDIA GPU** with CUDA (optional but strongly recommended — CPU works but is slow)

### Installation

```bash
# Clone the repository
git clone https://github.com/keyvanakhyani/EchoExtract.git
cd EchoExtract

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install the package and its dependencies
pip install -e .
```

> **GPU users (Windows):** the CUDA libraries are installed via pip:
> ```bash
> pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
> ```

### Usage

```bash
# Transcribe a video into all formats (output saved next to the video)
python -m echo_extract.cli "path/to/video.mp4"

# Choose specific output formats
python -m echo_extract.cli "path/to/video.mp4" -f srt -f json

# Use a smaller/faster model, run on CPU, and pick an output folder
python -m echo_extract.cli "path/to/video.mp4" -m small --device cpu -o "path/to/output"
```

| Option | Description | Default |
|---|---|---|
| `video` | Path to the input video (required) | — |
| `-f`, `--format` | Output format, repeatable: `txt`, `srt`, `vtt`, `json` | all four |
| `-o`, `--output` | Output folder | the video's own folder |
| `-m`, `--model` | Whisper model size | `large-v3` |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--keep-audio` | Keep the intermediate WAV file | off |

---

## 🛠️ Tech Stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — fast local Whisper inference
- [Pydantic](https://docs.pydantic.dev/) — typed domain models
- [ffmpeg](https://ffmpeg.org/) — audio extraction
- Python 3.10+ with full type hints

---

## 🗺️ Roadmap

- [ ] Optional input language override (`--language fa`)
- [ ] Language-based organization of outputs (Persian / English)
- [ ] Gradio web interface with i18n (Persian / English UI)
- [ ] Telegram bot for on-the-go transcription
- [ ] Optional LLM post-processing (summaries, cleanup) via OpenRouter

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
<br>

# EchoExtract (فارسی)

**تبدیل ویدیوهای فارسی و انگلیسی به متن، با استفاده از Whisper به‌صورت لوکال — و معماری تمیز و لایه‌ای.**

EchoExtract فایل‌های ویدیویی شما را به متن تبدیل می‌کند. صدا را استخراج می‌کند، آن را به یک مدل Whisper که به‌صورت محلی و روی GPU اجرا می‌شود می‌دهد، و متن را در چند فرمت مختلف می‌نویسد (متن ساده، زیرنویس SRT/VTT، و JSON با timestamp). زبان به‌صورت خودکار تشخیص داده می‌شود، پس هم فارسی و هم انگلیسی بدون تنظیم اضافه کار می‌کنند.

## ✨ امکانات

- 🎬 **ویدیو به متن** با یک دستور
- 🌍 **فارسی و انگلیسی** — تشخیص خودکار زبان (با مدل Whisper `large-v3`)
- 🖥️ **اجرای محلی روی GPU** — هیچ داده‌ای از سیستم شما خارج نمی‌شود و هزینه‌ی API ندارد
- 📄 **فرمت‌های خروجی متعدد** — `txt`، `srt`، `vtt`، `json` (هر کدام را که بخواهید)
- 🧩 **معماری engine قابل‌تعویض** — می‌توان backend را بدون تغییر بقیه‌ی برنامه عوض کرد
- 🧼 **کد تمیز، type-safe و لایه‌ای** — مدل‌های Pydantic، یک interface انتزاعی برای engine، و یک CLI سبک

## 🚀 شروع سریع

**پیش‌نیازها:** پایتون ۳.۱۰ به بالا، نصب ffmpeg، و ترجیحاً یک GPU انویدیا با CUDA.

```bash
git clone https://github.com/keyvanakhyani/EchoExtract.git
cd EchoExtract
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

**استفاده:**

```bash
# تبدیل یک ویدیو به همه‌ی فرمت‌ها (خروجی کنار خود ویدیو ساخته می‌شود)
python -m echo_extract.cli "path/to/video.mp4"

# انتخاب فرمت‌های خاص
python -m echo_extract.cli "path/to/video.mp4" -f srt -f json
```

## 📄 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است.