import os
import io
import wave
import hashlib
import sqlite3
import json
import base64
import numpy as np
from scipy import signal
from scipy.fft import fft, dct
from flask import Flask, request, jsonify, send_from_directory
import time

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'vocal.db')

# ─────────────────────────────────────────────
#  Database  (updated schema: multi-sample + pitch)
# ─────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            user_id          TEXT PRIMARY KEY,
            fingerprint_hash TEXT NOT NULL,
            feature_vector   TEXT NOT NULL,
            sample_count     INTEGER NOT NULL DEFAULT 1,
            created_at       INTEGER NOT NULL,
            updated_at       INTEGER NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS enrollment_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            features    TEXT NOT NULL,
            recorded_at INTEGER NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS signed_messages (
            msg_id        TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            watermark_key TEXT NOT NULL,
            audio_hash    TEXT NOT NULL,
            created_at    INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
#  Audio I/O
# ─────────────────────────────────────────────
def decode_wav_bytes(audio_bytes):
    buf = io.BytesIO(audio_bytes)
    try:
        with wave.open(buf, 'rb') as wf:
            sr         = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            n_frames   = wf.getnframes()
            raw        = wf.readframes(n_frames)
        fmt     = {1: 'b', 2: 'h', 4: 'i'}[sampwidth]
        samples = np.frombuffer(raw, dtype=np.dtype(fmt)).astype(np.float32)
        if n_channels == 2:
            samples = samples[::2]
        samples /= float(2 ** (8 * sampwidth - 1))
        return samples, sr
    except Exception:
        raise ValueError("Could not decode audio — ensure WAV format.")

def encode_wav_bytes(samples, sr):
    buf = io.BytesIO()
    samples_int = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())
    return buf.getvalue()

def resample_to(samples, src_sr, dst_sr=16000):
    if src_sr == dst_sr:
        return samples, dst_sr
    n_out = int(len(samples) * dst_sr / src_sr)
    resampled = signal.resample(samples, n_out)
    return resampled.astype(np.float32), dst_sr

# ─────────────────────────────────────────────
#  1. Voice Activity Detection (VAD)
# ─────────────────────────────────────────────
def apply_vad(samples, sr, frame_ms=20, energy_threshold_db=-35):
    frame_len  = int(sr * frame_ms / 1000)
    num_frames = len(samples) // frame_len
    active     = []
    for i in range(num_frames):
        frame = samples[i * frame_len:(i + 1) * frame_len]
        rms   = np.sqrt(np.mean(frame ** 2) + 1e-10)
        db    = 20 * np.log10(rms)
        if db > energy_threshold_db:
            active.append(frame)
    if not active:
        return samples
    return np.concatenate(active)

# ─────────────────────────────────────────────
#  2. Pitch (F0) Extraction
# ─────────────────────────────────────────────
def extract_pitch_features(samples, sr, frame_len=512, hop=256,
                            f0_min=60, f0_max=400):
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    f0_vals = []
    total_frames = 0
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len] * np.hanning(frame_len)
        corr  = np.correlate(frame, frame, mode='full')
        corr  = corr[len(corr) // 2:]
        corr /= (corr[0] + 1e-10)
        total_frames += 1
        if max_lag >= len(corr):
            continue
        segment  = corr[min_lag:max_lag]
        peak_idx = np.argmax(segment)
        peak_val = segment[peak_idx]
        if peak_val > 0.3:
            f0_vals.append(sr / (peak_idx + min_lag))
    if not f0_vals:
        return np.array([0.0, 0.0, 0.0])
    f0_arr          = np.array(f0_vals)
    voiced_fraction = len(f0_vals) / max(1, total_frames)
    return np.array([np.mean(f0_arr), np.std(f0_arr), voiced_fraction])

# ─────────────────────────────────────────────
#  3. Formant Estimation (F1, F2, F3)
# ─────────────────────────────────────────────
def estimate_formants(samples, sr, n_formants=3):
    lpc_order  = int(sr / 1000) + 2
    frame_len  = 512
    hop        = 256
    formants_per_frame = []
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len] * np.hanning(frame_len)
        r = np.array([np.dot(frame[:frame_len - k], frame[k:])
                      for k in range(lpc_order + 1)])
        if r[0] < 1e-10:
            continue
        # Levinson-Durbin
        a = np.zeros(lpc_order)
        e = r[0]
        for i in range(lpc_order):
            lam = (-sum(a[j] * r[i - j] for j in range(i)) - r[i + 1]) / (e + 1e-10)
            a_new  = a[:i] + lam * a[:i][::-1]
            a[i]   = lam
            a[:i]  = a_new
            e     *= (1 - lam ** 2)
        roots  = np.roots(np.concatenate([[1], a]))
        roots  = roots[np.imag(roots) > 0.01]
        angles = np.angle(roots)
        freqs  = sorted(angles * sr / (2 * np.pi))
        freqs  = [f for f in freqs if 90 < f < 4000]
        if len(freqs) >= n_formants:
            formants_per_frame.append(freqs[:n_formants])
    if not formants_per_frame:
        return np.zeros(n_formants * 2)
    arr = np.array(formants_per_frame)
    out = []
    for i in range(n_formants):
        out.extend([np.mean(arr[:, i]), np.std(arr[:, i])])
    return np.array(out)

# ─────────────────────────────────────────────
#  4. MFCC + Delta + Delta-Delta  (52 features)
# ─────────────────────────────────────────────
def compute_mfcc_full(samples, sr, n_mfcc=13, n_fft=512, hop=256):
    pre    = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    frames = np.array([
        pre[s:s + n_fft] * np.hanning(n_fft)
        for s in range(0, len(pre) - n_fft, hop)
    ])
    if frames.shape[0] < 3:
        return np.zeros(n_mfcc * 4)

    mag   = np.abs(fft(frames, axis=1))[:, :n_fft // 2 + 1]
    power = (1.0 / n_fft) * mag ** 2

    n_filters = 40
    high_mel  = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts   = np.linspace(0, high_mel, n_filters + 2)
    hz_pts    = 700 * (10 ** (mel_pts / 2595) - 1)
    bins      = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, mid, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, min(mid, fbank.shape[1])):
            if mid != lo: fbank[m-1, k] = (k - lo) / (mid - lo)
        for k in range(mid, min(hi, fbank.shape[1])):
            if hi != mid: fbank[m-1, k] = (hi - k) / (hi - mid)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)

    mfcc = np.zeros((frames.shape[0], n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(
            fb * np.cos(np.pi * n / n_filters * (np.arange(n_filters) + 0.5)),
            axis=1)

    # CMVN — removes mic/channel/room effects
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-10)

    delta  = np.diff(mfcc,  axis=0)
    delta2 = np.diff(delta, axis=0)

    return np.concatenate([
        np.mean(mfcc,   axis=0),   # 13
        np.std(mfcc,    axis=0),   # 13
        np.mean(delta,  axis=0),   # 13
        np.mean(delta2, axis=0),   # 13
    ])  # = 52

# ─────────────────────────────────────────────
#  5. Full Feature Vector: MFCC(52) + Pitch(3) + Formants(6) = 61
# ─────────────────────────────────────────────
def extract_full_features(samples, sr):
    samples, sr  = resample_to(samples, sr, dst_sr=16000)
    voiced       = apply_vad(samples, sr)
    mfcc_vec     = compute_mfcc_full(voiced, sr)
    pitch_vec    = extract_pitch_features(voiced, sr)
    formant_vec  = estimate_formants(voiced, sr)
    return np.concatenate([mfcc_vec, pitch_vec, formant_vec])  # 61

# ─────────────────────────────────────────────
#  6. Speaker Similarity — weighted multi-group score
# ─────────────────────────────────────────────
def speaker_similarity(enrolled_vec, query_vec):
    a = np.array(enrolled_vec, dtype=np.float64)
    b = np.array(query_vec,    dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    groups = [
        (slice(0,  13), 0.30),   # MFCC mean     — spectral shape
        (slice(13, 26), 0.15),   # MFCC std      — spectral variability
        (slice(26, 39), 0.20),   # Delta         — temporal dynamics
        (slice(39, 52), 0.15),   # Delta-delta   — speech acceleration
        (slice(52, 55), 0.12),   # Pitch (F0)    — vocal frequency
        (slice(55, 61), 0.08),   # Formants      — vocal tract resonance
    ]

    score = 0.0
    for sl, weight in groups:
        ga, gb = a[sl], b[sl]
        if len(ga) == 0:
            continue
        denom   = np.linalg.norm(ga) * np.linalg.norm(gb)
        cos_sim = float(np.dot(ga, gb) / denom) if denom > 0 else 0.0
        l2_dist  = np.linalg.norm(ga - gb) / np.sqrt(len(ga))
        l2_score = float(np.exp(-l2_dist / 8.0))
        score   += weight * (0.4 * cos_sim + 0.6 * l2_score)

    return float(np.clip(score, 0.0, 1.0))

# ─────────────────────────────────────────────
#  7. Multi-sample averaging
# ─────────────────────────────────────────────
def average_feature_vectors(vectors):
    return np.mean(np.array(vectors), axis=0)

def fingerprint_hash(feature_vec):
    return hashlib.sha256(
        json.dumps(np.round(feature_vec, 3).tolist()).encode()
    ).hexdigest()

# ─────────────────────────────────────────────
#  8. Watermark — DCT Domain (survives re-encoding)
# ─────────────────────────────────────────────
def _make_chip(key_seed, block_idx, n_coeffs):
    """Deterministic chip per block — same in embed and verify."""
    rng = np.random.default_rng(
        int(hashlib.sha256(f"{key_seed}:{block_idx}".encode()).hexdigest(), 16) % (2**32)
    )
    return rng.standard_normal(n_coeffs)

def embed_watermark_dct(samples, sr, secret_key):
    samples    = samples.copy().astype(np.float64)
    key_seed   = hashlib.sha256(secret_key.encode()).hexdigest()
    rng        = np.random.default_rng(int(key_seed, 16) % (2**32))
    block_size = 1024
    n_blocks   = len(samples) // block_size
    coeff_lo, coeff_hi = 100, 300
    strength   = 0.003
    pn_seq     = rng.choice([-1.0, 1.0], size=n_blocks)

    for i in range(n_blocks):
        s    = i * block_size
        D    = dct(samples[s:s + block_size], norm='ortho')
        chip = _make_chip(key_seed, i, coeff_hi - coeff_lo)
        D[coeff_lo:coeff_hi] += strength * pn_seq[i] * chip
        samples[s:s + block_size] = dct(D, type=3, norm='ortho')

    return np.clip(samples, -1.0, 1.0).astype(np.float32), pn_seq.tolist()

def verify_watermark_dct(samples, sr, secret_key, original_pn):
    samples    = samples.astype(np.float64)
    key_seed   = hashlib.sha256(secret_key.encode()).hexdigest()
    block_size = 1024
    n_blocks   = min(len(samples) // block_size, len(original_pn))
    coeff_lo, coeff_hi = 100, 300
    if n_blocks == 0:
        return 0.0

    pn_arr = np.array(original_pn[:n_blocks])
    scores = []
    for i in range(n_blocks):
        s      = i * block_size
        D      = dct(samples[s:s + block_size], norm='ortho')
        chip   = _make_chip(key_seed, i, coeff_hi - coeff_lo) * pn_arr[i]
        region = D[coeff_lo:coeff_hi]
        corr   = np.dot(region, chip) / (
            np.linalg.norm(region) * np.linalg.norm(chip) + 1e-10)
        scores.append(corr)

    return float(np.clip((np.mean(scores) + 1.0) / 2.0, 0.0, 1.0))

# ─────────────────────────────────────────────
#  Spectrogram helper
# ─────────────────────────────────────────────
def compute_spectrogram_data(samples, sr, n_fft=256):
    hop = n_fft // 2
    freqs, times, Sxx = signal.spectrogram(samples, sr, nperseg=n_fft, noverlap=hop)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    st = max(1, Sxx_db.shape[1] // 100)
    sf = max(1, Sxx_db.shape[0] // 50)
    return Sxx_db[::sf, ::st].tolist(), times[::st].tolist(), freqs[::sf].tolist()

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/enroll', methods=['POST'])
def enroll():
    try:
        data      = request.get_json()
        user_id   = data.get('user_id', '').strip()
        audio_b64 = data.get('audio')
        if not user_id or not audio_b64:
            return jsonify({'error': 'user_id and audio required'}), 400

        audio_bytes = base64.b64decode(audio_b64)
        samples, sr = decode_wav_bytes(audio_bytes)
        if len(samples) < sr * 1.0:
            return jsonify({'error': 'Recording too short — speak for at least 1 second'}), 400

        feat = extract_full_features(samples, sr)
        now  = int(time.time())

        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute('INSERT INTO enrollment_samples (user_id, features, recorded_at) VALUES (?,?,?)',
                  (user_id, json.dumps(feat.tolist()), now))
        c.execute('SELECT features FROM enrollment_samples WHERE user_id=?', (user_id,))
        all_feats = [json.loads(r[0]) for r in c.fetchall()]
        avg_feat  = average_feature_vectors(all_feats)
        fp_hash   = fingerprint_hash(avg_feat)
        c.execute('''INSERT OR REPLACE INTO fingerprints
                     (user_id, fingerprint_hash, feature_vector, sample_count, created_at, updated_at)
                     VALUES (?,?,?,?,COALESCE((SELECT created_at FROM fingerprints WHERE user_id=?),?),?)''',
                  (user_id, fp_hash, json.dumps(avg_feat.tolist()), len(all_feats), user_id, now, now))
        conn.commit()
        conn.close()

        spec, times, freqs = compute_spectrogram_data(samples, sr)
        return jsonify({
            'success':          True,
            'user_id':          user_id,
            'fingerprint_hash': fp_hash,
            'mfcc':             avg_feat[:13].tolist(),
            'pitch':            feat[52:55].tolist(),
            'formants':         feat[55:61].tolist(),
            'sample_count':     len(all_feats),
            'spectrogram':      spec,
            'times':            times,
            'freqs':            freqs,
            'duration':         len(samples) / sr,
            'sample_rate':      sr,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sign', methods=['POST'])
def sign():
    try:
        data      = request.get_json()
        user_id   = data.get('user_id', '').strip()
        audio_b64 = data.get('audio')
        if not user_id or not audio_b64:
            return jsonify({'error': 'user_id and audio required'}), 400

        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute('SELECT fingerprint_hash, feature_vector, sample_count FROM fingerprints WHERE user_id=?', (user_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'User not enrolled. Create a voice fingerprint first.'}), 404

        fp_hash, fv_json, sample_count = row
        enrolled_feat = json.loads(fv_json)

        audio_bytes   = base64.b64decode(audio_b64)
        samples, sr   = decode_wav_bytes(audio_bytes)
        msg_feat      = extract_full_features(samples, sr)
        speaker_score = speaker_similarity(enrolled_feat, msg_feat.tolist())

        msg_id     = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()[:16]
        secret_key = f"{fp_hash}:{msg_id}"
        watermarked, pn_seq = embed_watermark_dct(samples, sr, secret_key)
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute('INSERT INTO signed_messages (msg_id, user_id, watermark_key, audio_hash, created_at) VALUES (?,?,?,?,?)',
                  (msg_id, user_id, json.dumps({'key': secret_key, 'pn': pn_seq}), audio_hash, int(time.time())))
        conn.commit()
        conn.close()

        wm_b64 = base64.b64encode(encode_wav_bytes(watermarked, sr)).decode()
        spec, times, freqs = compute_spectrogram_data(watermarked, sr)
        return jsonify({
            'success':             True,
            'msg_id':              msg_id,
            'user_id':             user_id,
            'fingerprint_hash':    fp_hash,
            'audio_hash':          audio_hash,
            'speaker_match_score': round(speaker_score, 4),
            'enrollment_samples':  sample_count,
            'watermarked_audio':   wm_b64,
            'spectrogram':         spec,
            'times':               times,
            'freqs':               freqs,
            'signature':           f"{msg_id}:{fp_hash[:16]}",
            'duration':            len(samples) / sr,
            'sample_rate':         sr,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        data      = request.get_json()
        msg_id    = data.get('msg_id', '').strip()
        audio_b64 = data.get('audio')
        if not msg_id or not audio_b64:
            return jsonify({'error': 'msg_id and audio required'}), 400

        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute('SELECT user_id, watermark_key, audio_hash, created_at FROM signed_messages WHERE msg_id=?', (msg_id,))
        msg_row = c.fetchone()
        if not msg_row:
            conn.close()
            return jsonify({'error': 'Message ID not found in registry'}), 404

        user_id, wk_json, orig_hash, created_at = msg_row
        wk         = json.loads(wk_json)
        secret_key = wk['key']
        pn_seq     = wk['pn']

        c.execute('SELECT fingerprint_hash, feature_vector, sample_count FROM fingerprints WHERE user_id=?', (user_id,))
        fp_row = c.fetchone()
        conn.close()
        if not fp_row:
            return jsonify({'error': 'Enrolled fingerprint not found'}), 404

        fp_hash, fv_json, sample_count = fp_row
        enrolled_feat = json.loads(fv_json)

        audio_bytes   = base64.b64decode(audio_b64)
        samples, sr   = decode_wav_bytes(audio_bytes)
        wm_score      = verify_watermark_dct(samples, sr, secret_key, pn_seq)
        msg_feat      = extract_full_features(samples, sr)
        speaker_score = speaker_similarity(enrolled_feat, msg_feat.tolist())
        hash_match    = hashlib.sha256(audio_bytes).hexdigest() == orig_hash

        spec, times, freqs = compute_spectrogram_data(samples, sr)

        wm_pass      = wm_score      >= 0.62
        speaker_pass = speaker_score >= 0.52

        if wm_pass and speaker_pass:
            verdict = 'AUTHENTIC'
        elif wm_pass or speaker_pass:
            verdict = 'SUSPICIOUS'
        else:
            verdict = 'FAKE'

        # Per-component breakdown
        n = min(len(enrolled_feat), len(msg_feat.tolist()))
        ef = np.array(enrolled_feat[:n])
        mf = np.array(msg_feat.tolist()[:n])
        breakdown = {
            'mfcc_score':    round(speaker_similarity(ef[:52].tolist(), mf[:52].tolist()), 4),
            'pitch_score':   round(speaker_similarity(ef[52:55].tolist(), mf[52:55].tolist()), 4),
            'formant_score': round(speaker_similarity(ef[55:61].tolist(), mf[55:61].tolist()), 4),
        }

        return jsonify({
            'success':           True,
            'verdict':           verdict,
            'msg_id':            msg_id,
            'claimed_user':      user_id,
            'fingerprint_hash':  fp_hash,
            'enrollment_samples': sample_count,
            'watermark_score':   round(wm_score, 4),
            'speaker_score':     round(speaker_score, 4),
            'hash_match':        hash_match,
            'watermark_pass':    wm_pass,
            'speaker_pass':      speaker_pass,
            'breakdown':         breakdown,
            'enrolled_mfcc':     enrolled_feat[:13],
            'message_mfcc':      msg_feat[:13].tolist(),
            'spectrogram':       spec,
            'times':             times,
            'freqs':             freqs,
            'created_at':        created_at,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def list_users():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT user_id, fingerprint_hash, sample_count, updated_at FROM fingerprints ORDER BY updated_at DESC')
    rows = c.fetchall()
    conn.close()
    return jsonify([{
        'user_id':          r[0],
        'fingerprint_hash': r[1][:16] + '...',
        'sample_count':     r[2],
        'created_at':       r[3],
    } for r in rows])

@app.route('/api/messages', methods=['GET'])
def list_messages():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT msg_id, user_id, created_at FROM signed_messages ORDER BY created_at DESC LIMIT 20')
    rows = c.fetchall()
    conn.close()
    return jsonify([{'msg_id': r[0], 'user_id': r[1], 'created_at': r[2]} for r in rows])

@app.route('/api/enroll/count', methods=['GET'])
def enroll_count():
    user_id = request.args.get('user_id', '').strip()
    conn    = sqlite3.connect(DB_PATH)
    c       = conn.cursor()
    c.execute('SELECT sample_count FROM fingerprints WHERE user_id=?', (user_id,))
    row = c.fetchone()
    conn.close()
    return jsonify({'sample_count': row[0] if row else 0})

if __name__ == '__main__':
    app.run(debug=True, port=5050)
