# VocalPrint — Voice Authentication System

**Lightweight integrated voice authentication for real-world communication systems.**

## Setup & Run

```bash
python3 -m venv venv
venv\Scripts\activate
pip install flask numpy scipy
python app.py
# Open http://localhost:5050
```

## Architecture

```
vocal-fingerprint/
├── app.py          # Flask backend — DSP, watermarking, SQLite
├── index.html      # Single-file frontend — Web Audio API, visualizations
├── requirements.txt
└── db/
    └── vocal.db    # Auto-created SQLite database
```

## Features

### 1. Enroll (Voice Fingerprint)
- Record the standard phrase
- System computes 13-coefficient MFCC biometric signature
- SHA-256 hash stored in SQLite as the "voice fingerprint"
- Visual: spectrogram + MFCC bar chart

### 2. Sign (Watermark & Notarize)
- Record any voice message
- System embeds an **inaudible spread-spectrum watermark** near the Nyquist frequency (~18-20 kHz) using a PN sequence derived from your fingerprint key
- A unique **Message ID** is stored in the registry
- Download the watermarked WAV + share the Message ID

### 3. Verify (Dual Authentication)
- Upload/record the audio + enter the Message ID
- **Check 1 — Watermark:** Correlates embedded PN sequence → score ≥ 65% = PASS
- **Check 2 — Speaker:** MFCC cosine similarity vs enrolled fingerprint → score ≥ 75% = PASS
- **Verdict:** AUTHENTIC / SUSPICIOUS / FAKE

## Signal Processing Pipeline

```
ENROLLMENT:
  Audio → Pre-emphasis → Framing + Hanning Window
  → FFT Power Spectrum → 40-band Mel Filterbank
  → DCT → 13 MFCC Coefficients → SHA-256 Hash

SIGNING:
  Audio → MFCC (speaker identity check)
  → Generate PN sequence from fingerprint key
  → Embed spread-spectrum watermark @18-20 kHz
  → Store MSG_ID + PN in SQLite registry

VERIFICATION:
  Audio → Extract MFCC → Cosine sim vs enrolled fingerprint
  → Correlate with stored PN sequence
  → Dual pass/fail → AUTHENTIC / SUSPICIOUS / FAKE
```

## Research Gaps Addressed

1. **No practical end-user verification system** → Browser-based tool, no install needed
2. **Watermarking and speaker verification are separate** → Integrated dual-check
3. **No lightweight systems for web use** → Web Audio API + SciPy, no GPU required
4. **No digital signature concept for voice** → Voice digital signature framework
5. **Deepfake detection is reactive** → Proactive signing paradigm

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/enroll` | Create voice fingerprint |
| POST | `/api/sign` | Sign + watermark message |
| POST | `/api/verify` | Verify authenticity |
| GET | `/api/users` | List enrolled users |
| GET | `/api/messages` | List signed messages |
