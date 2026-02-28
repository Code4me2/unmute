"""Speaker gate: verify that the enrolled user is speaking before allowing barge-in.

Uses a WeSpeaker ONNX model directly via onnxruntime — no wespeaker Python
package needed.  Fbank feature extraction is implemented in pure numpy to
avoid pulling in torchaudio / heavy deps.
"""

from logging import getLogger
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = getLogger(__name__)

# Default ONNX model path (pre-downloaded in Docker build)
DEFAULT_MODEL_PATH = "/data/wespeaker.onnx"

# Fbank parameters (must match WeSpeaker training config)
_NUM_MEL_BINS = 80
_FRAME_LENGTH_MS = 25
_FRAME_SHIFT_MS = 10
_PREEMPH = 0.97


# ---------------------------------------------------------------------------
# Kaldi-compatible fbank feature extraction (pure numpy)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _mel_filterbank(num_bins: int, fft_size: int, sample_rate: int) -> np.ndarray:
    """Build a mel filterbank matrix [num_bins, fft_size // 2 + 1]."""
    num_fft_bins = fft_size // 2 + 1
    low_mel = _hz_to_mel(20.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fbank[i, j] = (right - j) / (right - center)
    return fbank


def compute_fbank(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = _NUM_MEL_BINS,
) -> np.ndarray:
    """Compute log-mel fbank features from raw PCM, Kaldi-compatible.

    Args:
        pcm: 1-D float32 waveform.
        sample_rate: Sample rate of *pcm*.
        num_mel_bins: Number of mel filter banks.

    Returns:
        2-D float32 array of shape ``(num_frames, num_mel_bins)``, with CMN
        applied.
    """
    # Scale to int16 range (Kaldi convention)
    wav = pcm.astype(np.float32) * (1 << 15)

    # Pre-emphasis
    wav = np.append(wav[0], wav[1:] - _PREEMPH * wav[:-1])

    frame_len = int(sample_rate * _FRAME_LENGTH_MS / 1000)
    frame_shift = int(sample_rate * _FRAME_SHIFT_MS / 1000)
    num_frames = 1 + (len(wav) - frame_len) // frame_shift
    if num_frames < 1:
        # Pad to at least one frame
        wav = np.pad(wav, (0, frame_len - len(wav)))
        num_frames = 1

    # Windowing (Povey window — like Hann but doesn't go to zero)
    window = np.power(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1)), 0.85).astype(np.float32)

    fft_size = 1
    while fft_size < frame_len:
        fft_size <<= 1

    mel_fb = _mel_filterbank(num_mel_bins, fft_size, sample_rate)

    features = np.empty((num_frames, num_mel_bins), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_shift
        frame = wav[start : start + frame_len] * window
        spectrum = np.abs(np.fft.rfft(frame, n=fft_size)) ** 2
        mel_energy = mel_fb @ spectrum
        mel_energy = np.maximum(mel_energy, 1e-10)
        features[i] = np.log(mel_energy)

    # Cepstral Mean Normalization
    features -= features.mean(axis=0)
    return features


class SpeakerGate:
    """Cosine-similarity speaker verification using a WeSpeaker ONNX model."""

    def __init__(
        self,
        enrollment_path: str | Path,
        threshold: float = 0.65,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=so)

        self.threshold = threshold
        self._enrollment = self._load_enrollment(Path(enrollment_path))
        logger.info(
            "[SpeakerGate] Loaded enrollment from %s (threshold=%.2f, model=%s)",
            enrollment_path,
            threshold,
            model_path,
        )

    @staticmethod
    def _load_enrollment(path: Path) -> np.ndarray:
        """Load a pre-computed .npy embedding."""
        if path.suffix != ".npy":
            raise ValueError(
                f"Enrollment must be a .npy file, got: {path.suffix}. "
                "Use enroll_user.py to create one."
            )
        emb = np.load(path).astype(np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def extract_embedding(self, pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a normalized embedding from raw PCM float32 audio.

        Args:
            pcm: 1-D float32 array at *sample_rate* Hz.
            sample_rate: Sample rate of the input audio.

        Returns:
            Normalized 1-D embedding vector.
        """
        feats = compute_fbank(pcm, sample_rate)
        # ONNX model expects (batch, frames, mel_bins)
        feats_input = feats[np.newaxis, :, :].astype(np.float32)
        outputs = self._session.run(
            ["embs"], {"feats": feats_input}
        )
        emb = outputs[0].flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def verify(self, pcm: np.ndarray, sample_rate: int = 16000) -> bool:
        """Return True if the speaker matches the enrolled profile."""
        return self.score(pcm, sample_rate) >= self.threshold

    def verify_with_score(
        self, pcm: np.ndarray, sample_rate: int = 16000
    ) -> tuple[bool, float]:
        """Return (match, cosine_score)."""
        s = self.score(pcm, sample_rate)
        return s >= self.threshold, s

    def score(self, pcm: np.ndarray, sample_rate: int = 16000) -> float:
        """Compute cosine similarity between pcm audio and enrollment."""
        emb = self.extract_embedding(pcm, sample_rate)
        return float(np.dot(emb, self._enrollment))

    def warmup(self) -> None:
        """Run a dummy inference to warm up the ONNX session."""
        dummy = np.zeros(16000, dtype=np.float32)
        try:
            self.extract_embedding(dummy, 16000)
        except Exception:
            pass  # Warm-up failures are non-fatal
        logger.info("[SpeakerGate] ONNX warm-up complete.")
