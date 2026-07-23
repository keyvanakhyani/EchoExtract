# EchoExtract

**Transcribe and translate Persian and English videos using local Whisper — GPU-accelerated, with a clean layered architecture.**

EchoExtract turns video files into text. It extracts audio, runs it through a locally-hosted Whisper model on your GPU, and writes transcripts in multiple formats. Language is auto-detected (or you can force it), and any supported language can additionally be translated to English.

---

## ✨ Features

- 🎬 **Video → text** in a single command
- 🌍 **99+ languages** with automatic detection — Persian and English work out of the box
- 🔤 **Translation to English** alongside the source-language transcript
- 🖥️ **Local GPU inference** — no data leaves your machine, no API costs
- ⚡ **Batched inference + VAD filtering** — roughly 2.5× faster than naive transcription, with no quality loss
- ⏱️ **Time-range selection** — transcribe just a slice of a long video
- 📄 **Four output formats** — `txt`, `srt`, `vtt`, `json` (pick any subset)
- 📊 **Live progress bar** with elapsed and remaining time estimates
- ⚙️ **Centralized configuration** via `pydantic-settings` and `.env` overrides
- 🧼 **Clean, typed, layered codebase** — Pydantic models, an abstract engine interface, and a thin CLI

---

## 🏗️ Architecture

```
src/echo_extract/
├── core/
│   ├── models.py          # Pydantic domain models (Segment, TranscriptionResult)
│   ├── config.py          # Centralized settings (pydantic-settings)
│   └── logging_config.py  # Rich-formatted application logging
├── engines/
│   ├── base.py            # Abstract TranscriptionEngine interface (Strategy pattern)
│   └── faster_whisper_engine.py   # Local GPU engine (faster-whisper / CTranslate2)
├── io/
│   ├── audio.py           # Audio extraction and trimming (ffmpeg)
│   └── writers.py         # Output writers with a format registry
├── pipeline.py            # Orchestration: video → audio → transcript → files
└── cli.py                 # Command-line interface
```

**Why this design?** The pipeline holds pure logic and knows nothing about *how* it's invoked. The CLI — and any future UI — simply calls `transcribe_video()`, so there's no duplicated logic. Engines implement a shared interface, so a cloud backend can be swapped in without touching the pipeline.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **ffmpeg** on your PATH ([download](https://ffmpeg.org/download.html))
- **NVIDIA GPU with CUDA** (optional but strongly recommended)

### Installation

```bash
git clone https://github.com/keyvanakhyani/EchoExtract.git
cd EchoExtract

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -e .
```

**For GPU acceleration**, install the CUDA runtime libraries:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The application locates these DLLs automatically, so no manual PATH setup is required.

---

## 📖 Usage

```bash
# Transcribe a video into all four formats (saved next to the video)
python -m echo_extract.cli "path/to/video.mp4"

# Persian video, subtitles only
python -m echo_extract.cli "lecture.mp4" -l fa -f srt

# Transcribe AND translate to English (produces two sets of files)
python -m echo_extract.cli "lecture.mp4" -l fa --translate-to en -f srt

# Transcribe just 3 minutes starting at the 5-minute mark
python -m echo_extract.cli "lecture.mp4" --start 300 --duration 180
```

### CLI Options

| Option | Short | Description | Default |
|---|---|---|---|
| `video` | — | Path to the input video (required) | — |
| `--format` | `-f` | Output format, repeatable: `txt`, `srt`, `vtt`, `json` | all four |
| `--output` | `-o` | Output folder | the video's own folder |
| `--language` | `-l` | Force source language (e.g. `fa`, `en`) | auto-detect |
| `--translate-to` | — | Also write a translation (currently `en` only) | off |
| `--model` | `-m` | Whisper model size | `large-v3` |
| `--device` | — | `cuda` or `cpu` | `cuda` |
| `--start` | — | Start time in seconds | beginning |
| `--duration` | — | Duration in seconds to process | whole video |
| `--keep-audio` | — | Keep the intermediate WAV file | off |
| `--verbose` | `-v` | Show detailed debug logs | off |

### Output Naming

Outputs are suffixed with their language, so transcripts and translations never collide:

```
lecture.fa.srt    # Persian transcript
lecture.en.srt    # English translation
```

---

## ⚙️ Configuration

All defaults live in `core/config.py` and can be overridden with environment variables or a `.env` file — no code changes needed:

```env
ECHO_MODEL_SIZE=large-v3
ECHO_DEVICE=cuda
ECHO_COMPUTE_TYPE=int8_float16
ECHO_BEAM_SIZE=5
ECHO_BATCH_SIZE=8
ECHO_HF_HOME=D:/whisper_models/huggingface
```

---

## 📊 Performance

Measured on an RTX 4060 Laptop (8 GB VRAM) with `large-v3`:

| Configuration | 3-minute audio |
|---|---|
| Sequential inference | ~36 s |
| **Batched inference + VAD** | **~15 s** |

Batched inference processes multiple audio chunks in parallel on the GPU, and VAD filtering skips silent regions — together roughly **2.5× faster with no reduction in transcription quality**.

---

## 🛠️ Tech Stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — local Whisper inference
- [Pydantic](https://docs.pydantic.dev/) & pydantic-settings — typed models and configuration
- [Rich](https://github.com/Textualize/rich) — progress bars and formatted logging
- [ffmpeg](https://ffmpeg.org/) — audio extraction
- Python 3.10+ with full type hints

---

## 🗺️ Roadmap

- [ ] Gradio web interface with Persian/English UI (i18n)
- [ ] Telegram bot for on-the-go transcription
- [ ] Chunked processing for resumable long-video runs
- [ ] LLM post-processing — summaries and translation to arbitrary languages
- [ ] Test suite

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---
<br>

# EchoExtract (فارسی)

**تبدیل ویدیوهای فارسی و انگلیسی به متن با Whisper محلی — با شتاب GPU و معماری تمیز و لایه‌ای.**

EchoExtract فایل‌های ویدیویی را به متن تبدیل می‌کند. صدا را استخراج می‌کند، آن را با مدل Whisper روی GPU پردازش می‌کند، و متن را در چند فرمت می‌نویسد. زبان به‌صورت خودکار تشخیص داده می‌شود (یا می‌توانید آن را مشخص کنید)، و امکان ترجمه به انگلیسی نیز وجود دارد.

## ✨ امکانات

- 🎬 **ویدیو به متن** با یک دستور
- 🌍 **بیش از ۹۹ زبان** با تشخیص خودکار — فارسی و انگلیسی بدون تنظیم اضافه
- 🔤 **ترجمه به انگلیسی** در کنار متن زبان اصلی
- 🖥️ **اجرای محلی روی GPU** — هیچ داده‌ای از سیستم شما خارج نمی‌شود
- ⚡ **پردازش دسته‌ای و فیلتر سکوت** — حدود ۲.۵ برابر سریع‌تر بدون افت کیفیت
- ⏱️ **انتخاب بازه‌ی زمانی** — پردازش بخشی از یک ویدیوی طولانی
- 📄 **چهار فرمت خروجی** — `txt`، `srt`، `vtt`، `json`
- 📊 **نوار پیشرفت زنده** با تخمین زمان باقی‌مانده
- ⚙️ **پیکربندی متمرکز** با `pydantic-settings` و پشتیبانی از `.env`

## 🚀 شروع سریع

**پیش‌نیازها:** پایتون ۳.۱۰ به بالا، ffmpeg، و ترجیحاً یک GPU انویدیا.

```bash
git clone https://github.com/keyvanakhyani/EchoExtract.git
cd EchoExtract
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 📖 نحوه‌ی استفاده

```bash
# تبدیل ویدیو به همه‌ی فرمت‌ها
python -m echo_extract.cli "path/to/video.mp4"

# ویدیوی فارسی، فقط زیرنویس
python -m echo_extract.cli "lecture.mp4" -l fa -f srt

# رونویسی به همراه ترجمه‌ی انگلیسی
python -m echo_extract.cli "lecture.mp4" -l fa --translate-to en -f srt

# پردازش فقط ۳ دقیقه، از دقیقه‌ی پنجم
python -m echo_extract.cli "lecture.mp4" --start 300 --duration 180
```

خروجی‌ها با پسوند زبان ذخیره می‌شوند تا متن اصلی و ترجمه با هم قاطی نشوند:

```
lecture.fa.srt    # متن فارسی
lecture.en.srt    # ترجمه‌ی انگلیسی
```

## 📊 کارایی

روی RTX 4060 لپ‌تاپی (۸ گیگابایت VRAM) با مدل `large-v3`:

| پیکربندی | صدای ۳ دقیقه‌ای |
|---|---|
| پردازش ترتیبی | حدود ۳۶ ثانیه |
| **پردازش دسته‌ای + VAD** | **حدود ۱۵ ثانیه** |

## 📄 لایسنس

MIT