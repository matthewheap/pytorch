"""
Vocoder Web App — Flask backend with PyTorch audio processing.
Records mic input, processes with torchaudio, returns layered audio buffers.
"""

import io
import base64

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
from flask import Flask, render_template, request, jsonify
from scipy.io import wavfile as scipy_wav

app = Flask(__name__)

# ---------------------------------------------------------------------------
# PyTorch Autoencoder for timbral variation
# ---------------------------------------------------------------------------

class AudioAutoencoder(nn.Module):
    """Small 1-D convolutional autoencoder that creates timbral variations."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, latent_dim, kernel_size=15, stride=2, padding=7),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# Lazy-load autoencoder to avoid issues with Flask debug reloader
_autoencoder = None

def get_autoencoder() -> AudioAutoencoder:
    """Get or create the autoencoder instance (lazy singleton)."""
    global _autoencoder
    if _autoencoder is None:
        _autoencoder = AudioAutoencoder()
        _autoencoder.eval()
    return _autoencoder

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def wav_bytes_to_tensor(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Convert raw WAV bytes to a (1, N) float tensor and sample rate."""
    buf = io.BytesIO(wav_bytes)
    sr, data = scipy_wav.read(buf)
    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    # Mix to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    return waveform, sr


def tensor_to_wav_base64(waveform: torch.Tensor, sr: int) -> str:
    """Encode a (1, N) float tensor as a base64 WAV string."""
    samples = waveform.squeeze(0).clamp(-1, 1).numpy()
    int_samples = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy_wav.write(buf, sr, int_samples)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_features(waveform: torch.Tensor, sr: int) -> dict:
    """Extract spectral features using PyTorch / torchaudio."""
    # MFCC
    mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13,
                             melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64})
    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=1024,
                                      hop_length=512, n_mels=64)
    mel_spec = mel_transform(waveform)  # (1, n_mels, time)

    # Spectral centroid (weighted mean of frequencies)
    freqs = torch.linspace(0, sr / 2, mel_spec.shape[1])
    spec_sum = mel_spec.sum(dim=1, keepdim=True).clamp(min=1e-8)
    spectral_centroid = (mel_spec * freqs.unsqueeze(0).unsqueeze(-1)).sum(dim=1) / spec_sum.squeeze(1)
    avg_centroid = spectral_centroid.mean().item()

    # Energy / RMS
    rms = waveform.pow(2).mean().sqrt().item()

    # Zero crossing rate
    signs = torch.sign(waveform)
    zcr = ((signs[:, 1:] - signs[:, :-1]).abs() / 2).mean().item()

    # Dominant frequency from FFT
    fft_mag = torch.abs(torch.fft.rfft(waveform))
    dominant_bin = fft_mag.argmax(dim=-1).item()
    dominant_freq = dominant_bin * sr / waveform.shape[-1]

    return {
        "spectral_centroid": round(avg_centroid, 2),
        "rms_energy": round(rms, 5),
        "zero_crossing_rate": round(zcr, 5),
        "dominant_frequency": round(dominant_freq, 2),
        "mfcc_mean": mfcc.mean(dim=-1).squeeze().tolist(),
        "mel_energy_profile": mel_spec.mean(dim=-1).squeeze().tolist(),
    }


def detect_fundamental_frequency(waveform: torch.Tensor, sr: int) -> float:
    """
    Detect fundamental frequency using PyTorch autocorrelation.
    More accurate than simple FFT peak for voiced sounds.
    """
    signal = waveform.squeeze(0)
    # High-pass filter to remove DC offset
    signal = signal - signal.mean()

    # Autocorrelation via FFT
    n = signal.shape[0]
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    padded = torch.zeros(fft_size)
    padded[:n] = signal
    spectrum = torch.fft.rfft(padded)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj())
    autocorr = autocorr[:n]

    # Normalize
    autocorr = autocorr / autocorr[0].clamp(min=1e-8)

    # Find first peak after the initial decay
    # Skip lags corresponding to frequencies above 1000 Hz
    min_lag = max(1, int(sr / 1000))
    # Don't look for frequencies below 50 Hz
    max_lag = min(n - 1, int(sr / 50))

    if min_lag >= max_lag:
        return 0.0

    search = autocorr[min_lag:max_lag]
    if search.numel() == 0:
        return 0.0

    peak_idx = search.argmax().item() + min_lag
    peak_val = autocorr[peak_idx].item()

    # Confidence check: autocorrelation peak should be significant
    if peak_val < 0.2:
        return 0.0

    fundamental_hz = sr / peak_idx
    return round(fundamental_hz, 2)


def compute_amplitude_envelope(waveform: torch.Tensor, sr: int,
                                hop_ms: float = 30.0) -> list[float]:
    """
    Compute amplitude envelope using windowed RMS.
    Returns a list of RMS values, one per hop window.
    Used to make partials pulse in time with the voice.
    """
    hop_samples = int(sr * hop_ms / 1000)
    signal = waveform.squeeze(0).abs()
    n = signal.shape[0]
    envelope = []

    for start in range(0, n, hop_samples):
        end = min(start + hop_samples, n)
        chunk = signal[start:end]
        rms = chunk.pow(2).mean().sqrt().item()
        envelope.append(round(rms, 6))

    # Normalize envelope to 0-1 range
    if envelope:
        max_val = max(envelope) if max(envelope) > 0 else 1.0
        envelope = [round(v / max_val, 4) for v in envelope]

    return envelope


def create_grain(waveform: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Extract a grain with Hann window applied."""
    end = min(start + length, waveform.shape[-1])
    grain = waveform[:, start:end].clone()
    window = torch.hann_window(grain.shape[-1], device=grain.device)
    return grain * window


def pitch_shift_layer(waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    """Pitch-shift a waveform by the given number of semitones using torchaudio."""
    shifted = F.pitch_shift(waveform, sr, semitones)
    return shifted


def detect_onsets(waveform: torch.Tensor, sr: int, hop_length: int = 512) -> torch.Tensor:
    """
    Detect onset positions using PyTorch spectral flux.
    Returns a 1-D tensor of sample indices where onsets occur.
    """
    # Compute STFT magnitude
    n_fft = 1024
    spec = torch.stft(waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=torch.hann_window(n_fft),
                      return_complex=True)
    mag = spec.abs()  # (freq_bins, time_frames)

    # Spectral flux: sum of positive differences between consecutive frames
    diff = mag[:, 1:] - mag[:, :-1]
    flux = torch.relu(diff).sum(dim=0)  # (time_frames - 1,)

    # Adaptive threshold: mean + 1.5 * std
    threshold = flux.mean() + 1.5 * flux.std()
    onset_frames = torch.where(flux > threshold)[0]

    # Remove onsets that are too close together (within 50ms)
    min_gap = int(0.05 * sr / hop_length)
    if len(onset_frames) > 1:
        filtered = [onset_frames[0]]
        for f in onset_frames[1:]:
            if f - filtered[-1] >= min_gap:
                filtered.append(f)
        onset_frames = torch.tensor(filtered)

    # Convert frame indices to sample indices
    onset_samples = onset_frames * hop_length
    return onset_samples


def extract_transients(waveform: torch.Tensor, sr: int, onset_samples: torch.Tensor,
                       transient_dur: float = 0.08) -> list[torch.Tensor]:
    """
    Extract short transient slices around each onset point.
    Each transient is windowed for smooth edges.
    """
    transient_len = int(sr * transient_dur)
    total = waveform.shape[-1]
    transients = []

    for onset in onset_samples:
        start = max(0, int(onset.item()) - transient_len // 4)  # slight pre-onset
        end = min(total, start + transient_len)
        if end - start < transient_len // 2:
            continue
        t = waveform[:, start:end].clone()
        window = torch.hann_window(t.shape[-1], device=t.device)
        transients.append(t * window)

    return transients


def build_rhythmic_layer(transients: list[torch.Tensor], sr: int,
                         duration_samples: int, pattern: str = "straight") -> torch.Tensor:
    """
    Sequence extracted transients into a rhythmic pattern.
    Patterns: 'straight' (even spacing), 'shuffle' (swing feel),
              'stutter' (rapid fire), 'polyrhythm' (5-over-4 feel).
    """
    if not transients:
        return torch.zeros(1, duration_samples)

    output = torch.zeros(1, duration_samples)

    if pattern == "straight":
        # Even 8th-note spacing, auto-detect a reasonable tempo
        num_hits = max(8, duration_samples // (sr // 8))  # ~8 hits per second
        spacing = duration_samples // num_hits
        for i in range(num_hits):
            t = transients[i % len(transients)]
            pos = i * spacing
            end = min(pos + t.shape[-1], duration_samples)
            seg = end - pos
            output[:, pos:end] += t[:, :seg] * (0.6 + 0.4 * int((i % 4) == 0))  # accent downbeats

    elif pattern == "shuffle":
        # Swing feel: alternating long-short gaps
        beat_dur = sr // 4  # quarter note at ~240bpm feel
        pos = 0
        i = 0
        while pos < duration_samples:
            t = transients[i % len(transients)]
            end = min(pos + t.shape[-1], duration_samples)
            seg = end - pos
            gain = 0.8 if i % 2 == 0 else 0.5  # accent on-beats
            output[:, pos:end] += t[:, :seg] * gain
            # Swing ratio: ~67/33
            gap = int(beat_dur * 0.67) if i % 2 == 0 else int(beat_dur * 0.33)
            pos += gap
            i += 1

    elif pattern == "stutter":
        # Rapid-fire repetitions with decreasing volume
        stutter_gap = sr // 16  # 16th notes at fast tempo
        pos = 0
        i = 0
        while pos < duration_samples:
            t = transients[i % len(transients)]
            # Every 4th hit, switch to a different transient
            if i % 4 == 0 and len(transients) > 1:
                t = transients[(i // 4) % len(transients)]
            end = min(pos + t.shape[-1], duration_samples)
            seg = end - pos
            decay = 0.9 ** (i % 8)  # decay within each group of 8
            output[:, pos:end] += t[:, :seg] * decay * 0.7
            pos += stutter_gap
            i += 1

    elif pattern == "polyrhythm":
        # Layer 5-over-4 polyrhythm
        beat_5 = duration_samples // 5
        beat_4 = duration_samples // 4
        for i in range(5):
            t = transients[0]
            pos = i * beat_5
            end = min(pos + t.shape[-1], duration_samples)
            output[:, pos:end] += t[:, :end - pos] * 0.7
        for i in range(4):
            t = transients[min(1, len(transients) - 1)]
            pos = i * beat_4
            end = min(pos + t.shape[-1], duration_samples)
            output[:, pos:end] += t[:, :end - pos] * 0.5

    # Normalize
    peak = output.abs().max().clamp(min=1e-6)
    output = output / peak * 0.7
    return output


# ---------------------------------------------------------------------------
# Layer generation pipeline
# ---------------------------------------------------------------------------

def generate_layers(waveform: torch.Tensor, sr: int, num_layers: int = 6) -> list[dict]:
    """
    Generate multiple audio layers from the input waveform:
      - Original (clean)
      - Pitch-shifted variants (+/- semitones)
      - Autoencoder timbral variations
      - Granular re-synthesis layer
      - Reversed layer
      - Rhythmic Hits (onset-extracted transients in a pattern)
      - Rhythmic Stutter (rapid micro-sliced transients)
    """
    layers = []
    duration_samples = waveform.shape[-1]

    # Normalize input
    peak = waveform.abs().max().clamp(min=1e-6)
    waveform = waveform / peak

    # 1. Original layer
    layers.append({
        "name": "Original",
        "audio": tensor_to_wav_base64(waveform, sr),
        "default_pan": 0.0,
        "default_gain": 0.8,
    })

    # 2-3. Pitch-shifted layers
    pitch_offsets = [-5, -12, 7, 12, -7, 3]
    pitch_count = min(2, num_layers - 1)
    for i, semitones in enumerate(pitch_offsets[:pitch_count]):
        shifted = pitch_shift_layer(waveform, sr, semitones)
        pan = -0.7 if i % 2 == 0 else 0.7
        layers.append({
            "name": f"Pitch {'+' if semitones > 0 else ''}{semitones}st",
            "audio": tensor_to_wav_base64(shifted, sr),
            "default_pan": pan,
            "default_gain": 0.4,
        })

    # 4. Autoencoder timbral variation
    if num_layers > 3:
        with torch.no_grad():
            # Pad to multiple of 16 for conv layers
            pad_len = (16 - (duration_samples % 16)) % 16
            padded = torch.nn.functional.pad(waveform.unsqueeze(0), (0, pad_len))
            variation = get_autoencoder().forward(padded).squeeze(0)
            variation = variation[:, :duration_samples]
            # Normalize
            var_peak = variation.abs().max().clamp(min=1e-6)
            variation = variation / var_peak * 0.8
            # Shift down one octave to soften harsh high-frequency content
            variation = pitch_shift_layer(variation, sr, -12)
            # Low-pass filter: remove everything above 1200 Hz
            variation = F.lowpass_biquad(variation, sr, cutoff_freq=1200.0)
        layers.append({
            "name": "Neural Texture",
            "audio": tensor_to_wav_base64(variation, sr),
            "default_pan": -0.3,
            "default_gain": 0.35,
        })

    # 5. Granular re-synthesis layer
    if num_layers > 4:
        grain_size = int(sr * 0.15)  # 150ms grains
        num_grains = max(1, duration_samples // grain_size)
        granular = torch.zeros(1, duration_samples)
        for g in range(num_grains):
            offset = torch.randint(0, max(1, duration_samples - grain_size), (1,)).item()
            grain = create_grain(waveform, offset, grain_size)
            place = int(g * grain_size * 0.7)
            end = min(place + grain.shape[-1], duration_samples)
            seg_len = end - place
            granular[:, place:end] += grain[:, :seg_len]
        gran_peak = granular.abs().max().clamp(min=1e-6)
        granular = granular / gran_peak * 0.6
        layers.append({
            "name": "Granular",
            "audio": tensor_to_wav_base64(granular, sr),
            "default_pan": 0.5,
            "default_gain": 0.3,
        })

    # 6. Reversed layer
    if num_layers > 5:
        reversed_wave = waveform.flip(dims=[-1])
        layers.append({
            "name": "Reversed",
            "audio": tensor_to_wav_base64(reversed_wave, sr),
            "default_pan": -0.5,
            "default_gain": 0.25,
        })

    # --- Rhythmic layers via onset detection ---
    onset_samples = detect_onsets(waveform, sr)
    transients = extract_transients(waveform, sr, onset_samples)

    # 7. Rhythmic Hits — extracted transients in a straight pattern
    if num_layers > 6 and transients:
        rhythmic = build_rhythmic_layer(transients, sr, duration_samples, pattern="straight")
        layers.append({
            "name": "Rhythmic Hits",
            "audio": tensor_to_wav_base64(rhythmic, sr),
            "default_pan": 0.3,
            "default_gain": 0.45,
        })

    # 8. Rhythmic Stutter — rapid micro-slice pattern
    if num_layers > 7 and transients:
        stutter = build_rhythmic_layer(transients, sr, duration_samples, pattern="stutter")
        layers.append({
            "name": "Rhythmic Stutter",
            "audio": tensor_to_wav_base64(stutter, sr),
            "default_pan": -0.4,
            "default_gain": 0.35,
        })

    # 9. Shuffle Groove
    if num_layers > 8 and transients:
        shuffle = build_rhythmic_layer(transients, sr, duration_samples, pattern="shuffle")
        layers.append({
            "name": "Shuffle Groove",
            "audio": tensor_to_wav_base64(shuffle, sr),
            "default_pan": 0.6,
            "default_gain": 0.3,
        })

    # 10. Polyrhythm
    if num_layers > 9 and transients:
        poly = build_rhythmic_layer(transients, sr, duration_samples, pattern="polyrhythm")
        layers.append({
            "name": "Polyrhythm",
            "audio": tensor_to_wav_base64(poly, sr),
            "default_pan": -0.6,
            "default_gain": 0.3,
        })

    # Fill remaining slots with extra pitch shifts if needed
    extra_idx = 2
    while len(layers) < num_layers and extra_idx < len(pitch_offsets):
        semitones = pitch_offsets[extra_idx]
        shifted = pitch_shift_layer(waveform, sr, semitones)
        pan = 0.4 if extra_idx % 2 == 0 else -0.4
        layers.append({
            "name": f"Pitch {'+' if semitones > 0 else ''}{semitones}st",
            "audio": tensor_to_wav_base64(shifted, sr),
            "default_pan": pan,
            "default_gain": 0.3,
        })
        extra_idx += 1

    return layers[:num_layers]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_audio():
    """Accept a recorded WAV, run PyTorch pipeline, return layers + features."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    wav_bytes = audio_file.read()

    if len(wav_bytes) < 100:
        return jsonify({"error": "Audio file too small"}), 400

    try:
        waveform, sr = wav_bytes_to_tensor(wav_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not decode audio: {str(e)}"}), 400

    # Extract features with PyTorch
    features = compute_features(waveform, sr)

    # Get requested layer count (default 6)
    num_layers = int(request.form.get("num_layers", 6))
    num_layers = max(1, min(10, num_layers))

    # Generate audio layers
    layers = generate_layers(waveform, sr, num_layers)

    # Detect fundamental frequency and amplitude envelope for partials
    fundamental_hz = detect_fundamental_frequency(waveform, sr)
    envelope = compute_amplitude_envelope(waveform, sr, hop_ms=30.0)

    return jsonify({
        "features": features,
        "layers": layers,
        "sample_rate": sr,
        "duration": waveform.shape[-1] / sr,
        "fundamental_hz": fundamental_hz,
        "envelope": envelope,
        "envelope_hop_ms": 30.0,
    })


if __name__ == "__main__":
    print(" Vocoder Web App starting...")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
