# ExplainabilIty-AI

A **universal multimodal explainable AI framework** built with Streamlit.

This project combines model-process-level explainability for text, images, audio, video, and documents with a separate **Explainable Travel Agent** mode. It also includes a **Diffusion Lab** for generating images while capturing denoising-step traces. The app is wired around Gemini for human-readable explanations and uses a broad XAI toolchain across NLP, vision, audio, video, and document processing.

## Features

- **Multimodal XAI dashboard**
  - Text explainability
  - Image explainability
  - Audio explainability
  - Video explainability
  - Document explainability for **PDF, DOCX, and PPTX**
- **Explainable Travel Agent**
  - Generates travel plans
  - Shows a decision trace for agent/tool calls
  - Explains why each step was chosen
- **Diffusion Lab**
  - In-app image generation
  - Denoising-step tracing
  - Model selection support
- **Human-readable explanations**
  - Gemini-backed narration
  - Mechanistic XAI style explanations
- **Performance options**
  - Fast mode for long PDFs/documents
  - Optional SHAP token attribution for text
  - Optional CLIP prompt-region heatmaps for images

## Tech Stack

- **Frontend:** Streamlit
- **Core ML / AI:** PyTorch, Transformers, Sentence-Transformers, scikit-learn
- **XAI / Attribution:** SHAP, LIME, Captum
- **Generative AI:** Google Gemini
- **Diffusion:** Diffusers
- **Document processing:** PyMuPDF, python-docx, python-pptx
- **Audio / video:** librosa, soundfile, moviepy, OpenCV, Whisper
- **Utilities:** NLTK, NumPy, Pillow, matplotlib

## Installation

```bash
git clone https://github.com/Aryan-Rajesh-Python/ExplainabilIty-AI.git
cd ExplainabilIty-AI
```
## Activating Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```
## Installing Requirements

```bash
pip install -r requirements.txt
```
## Configuring API

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key"

# macOS / Linux
export GEMINI_API_KEY="your_api_key"
```
## How To Run

```bash
streamlit run app.py
