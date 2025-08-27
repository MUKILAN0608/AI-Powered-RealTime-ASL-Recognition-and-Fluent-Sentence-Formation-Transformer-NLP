# Sign Language Recognition – Real‑time ASL to Natural English

A real‑time sign language recognition application that combines MediaPipe hand tracking, a trained classifier, robust spell/word correction, and safe sentence framing. The UI is intentionally minimal and professional: only the current word and formed sentence appear at the top‑left; all processing happens in the background.

## Key Capabilities

- Real‑time hand landmark tracking (MediaPipe)
- Robust letter → word formation with temporal stability
- Strong word correction pipeline (spellcheck + typo‑aware similarity)
- POS‑aware agreement (I am / he is / we are) during local framing
- Safe sentence formation: uses a local framer or Ollama strictly for arranging words (never changing them)
- Natural TTS output in the background

## Requirements

- Python 3.8+
- A webcam
- Model file: `sign_language_model.pkl`

Install dependencies:
```bash
pip install -r requirements.txt
```

Optional: place `frequency_dict.txt` in the project root to expand the spellcheck vocabulary.

## Quick Start

```bash
python completed.py
```

Controls (kept minimal on screen):
- SPACE: add current word to the sentence
- ENTER: frame and speak the sentence
- BACKSPACE: remove last word
- C: clear sentence
- Q: quit

## How It Works

1) Tracking and letters
- Live frames → MediaPipe hands → 21‑point landmarks → classifier → letters
- Temporal stability ensures a letter is appended only after it is steady

2) Word correction (preserves intent)
- If a token exists in the spellcheck set, it is kept
- Otherwise we correct using a layered strategy:
  - First/last‑letter‑aware nearest‑word search (Levenshtein + keyboard‑typo similarity)
  - Repeated‑letter tolerance (cooool → cool)
  - Fallback to curated dictionaries (advanced/common)
  - Examples: kikl → kill, fkee → free, bild → build, hapyy → happy

3) Sentence formation (no hallucination)
- Words are framed into a sentence; we enforce subject‑verb agreement (to‑be: am/is/are) and punctuation
- Ollama can be used to arrange words more naturally, but the system verifies that ALL corrected words are preserved; if not, it falls back to local framing

4) Output
- The predicted word and the current sentence are rendered in the top‑left
- The final sentence is spoken via TTS

## Configuration Highlights

- Camera: 1920×1080, 30 FPS (defaults in code)
- Recognition stability: ~2 seconds hold per letter (tunable)
- Spellcheck sources: `frequency_dict.txt`, validator common words, correction dictionaries
- Window title: “Sign Language Recognition”

## Troubleshooting

- Camera not available: close other apps using the webcam; check permissions
- Model missing: ensure `sign_language_model.pkl` is in the project root
- TTS issues: verify system audio; try a different voice/rate in code if needed
- Corrections feel off: add domain terms to `frequency_dict.txt` to bias the spellcheck

## Repository Layout (excerpt)

```
completed.py            # Main application
requirements*.txt       # Dependency lists
sign_language_model.pkl # Trained classifier (required)
frequency_dict.txt      # Optional spell list for correction
logs/recognition.log    # Runtime log
```

## License

MIT License. See LICENSE for details.

## Acknowledgments

- MediaPipe Hands for landmark detection
- Community NLP resources for spellchecking and agreement cues

---

For best results, ensure good lighting and keep hands within the camera frame. The application will display the predicted word and the evolving sentence at the top‑left while speaking the finalized sentence.
