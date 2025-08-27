# 🎛️ BART‑T5‑LLama3‑Enhanced‑ASL‑Translation‑with‑Linguistic‑Refinement‑and‑Speech‑Synthesis

An end‑to‑end ASL-to-English system that blends real‑time visual recognition, robust linguistic refinement (spellcheck + grammar + POS agreement), safe sentence formation (local and LLama3‑guarded), and natural speech synthesis — all wrapped in a clean, professional UI.

## ✨ Key Capabilities

- 🖐️ Real‑time hand landmark tracking (MediaPipe)
- 🔤 Letter ➜ word formation with temporal stability (debounce/hold detection)
- 🧠 Correction pipeline: spellcheck + typo‑aware similarity (Levenshtein, keyboard proximity, repeat tolerance)
- 🧾 POS‑aware subject‑verb agreement (I am / he is / we are)
- 🧩 Safe sentence formation: local framer or LLama3 (Ollama) strictly for arranging words; words are preserved
- 🔈 Background text‑to‑speech (queued) for fluent audio output
- 🧼 Minimal, distraction‑free UI: predicted word and formed sentence at top‑left

## 📦 Requirements

- Python 3.8+
- A webcam
- Model file: `sign_language_model.pkl`

Install dependencies:
```bash
pip install -r requirements.txt
```

Optional: place `frequency_dict.txt` in the project root to expand the spellcheck vocabulary.

## 🚀 Quick Start

```bash
python completed.py
```

Controls (kept minimal on screen):
- SPACE: add current word to the sentence
- ENTER: frame and speak the sentence
- BACKSPACE: remove last word
- C: clear sentence
- Q: quit

## 🧭 How It Works

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

## ⚙️ Configuration Highlights

- Camera: 1920×1080, 30 FPS (defaults in code)
- Recognition stability: ~2 seconds hold per letter (tunable)
- Spellcheck sources: `frequency_dict.txt`, validator common words, correction dictionaries
- Window title: “Sign Language Recognition”

## 🧩 Detailed NLP/Linguistics Pipeline

- Token acceptance policy: keep exact tokens that already exist in the spellcheck set
- Unknown tokens: correct via combined score = Levenshtein + keyboard proximity + repeat collapse, with first/last letter preference and length windowing
- POS layer: spaCy‑based agreement pass for to‑be verbs (am/is/are) and capitalization of “I”
- Sentence formation safety: LLama3 is prompt‑constrained to arrange words, not invent them; output is rejected if any corrected word is missing, then local framing is applied

## 🧪 Examples

- Input letters: B I L L → word “BILL” → correction “build” (dictionary and similarity)
- Input: “i are happy” → agreement → “I am happy.”
- Input noisy token “fkee” → corrected to “free” → included verb phrase becomes “I am free.” when framed

## 🛠️ Troubleshooting

- Camera not available: close other apps using the webcam; check permissions
- Model missing: ensure `sign_language_model.pkl` is in the project root
- TTS issues: verify system audio; try a different voice/rate in code if needed
- Corrections feel off: add domain terms to `frequency_dict.txt` to bias the spellcheck

## 📁 Repository Layout (excerpt)

```
completed.py            # Main application
requirements*.txt       # Dependency lists
sign_language_model.pkl # Trained classifier (required)
frequency_dict.txt      # Optional spell list for correction
logs/recognition.log    # Runtime log
```

## 📄 License

MIT License. See LICENSE for details.

## 🙏 Acknowledgments

- MediaPipe Hands for landmark detection
- Community NLP resources for spellchecking and agreement cues

---

For best results, ensure good lighting and keep hands within the camera frame. The application will display the predicted word and the evolving sentence at the top‑left while speaking the finalized sentence.
