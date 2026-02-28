#!/usr/bin/env python3
"""CLI tool for enrolling a speaker profile for the speaker gate.

Usage:
    # Record 10 seconds from mic and save enrollment:
    python enroll_user.py --record 10 --output enrollment/user_profile.npy

    # Enroll from an existing WAV file:
    python enroll_user.py --wav my_voice.wav --output enrollment/user_profile.npy

    # Test an enrollment against a WAV file:
    python enroll_user.py --test enrollment/user_profile.npy --wav test_clip.wav

    # Merge multiple enrollment files (averaging):
    python enroll_user.py --merge enrollment/a.npy enrollment/b.npy --output enrollment/merged.npy

Requires: onnxruntime, soundfile, numpy.  Optional: sounddevice (for --record), soxr (for resampling).
The ONNX model must be downloaded first — see the Dockerfile or run:
    wget -O wespeaker.onnx 'https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet34_LM.onnx'
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice is required for recording. Install with: pip install sounddevice")
        sys.exit(1)

    print(f"Recording {duration:.1f}s of audio... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def load_audio(wav_path: str) -> np.ndarray:
    """Load a WAV file and return 16kHz mono float32."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        import soxr
        audio = soxr.resample(audio, sr, 16000)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker profile for the speaker gate.")
    parser.add_argument("--wav", type=str, help="Path to a WAV file to enroll from")
    parser.add_argument("--record", type=float, metavar="SECONDS", help="Record from mic for N seconds")
    parser.add_argument("--output", "-o", type=str, default="enrollment/user_profile.npy", help="Output path for the .npy enrollment file")
    parser.add_argument("--test", type=str, metavar="ENROLLMENT_NPY", help="Test an enrollment against --wav")
    parser.add_argument("--merge", nargs="+", metavar="NPY", help="Merge multiple .npy enrollments by averaging")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity threshold for --test (default: 0.65)")
    parser.add_argument("--model", type=str, default="wespeaker.onnx", help="Path to the WeSpeaker ONNX model")

    args = parser.parse_args()

    if args.merge:
        embeddings = [np.load(p) for p in args.merge]
        merged = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, merged)
        print(f"Merged {len(embeddings)} enrollments -> {out_path}")
        return

    if args.test:
        if not args.wav:
            parser.error("--test requires --wav to compare against")

        from unmute.speaker_gate import SpeakerGate
        gate = SpeakerGate(args.test, threshold=args.threshold, model_path=args.model)

        audio = load_audio(args.wav)
        match, score = gate.verify_with_score(audio, 16000)
        status = "MATCH" if match else "NO MATCH"
        print(f"Score: {score:.4f}  Threshold: {args.threshold:.2f}  -> {status}")
        return

    # Enrollment mode
    if args.wav:
        audio = load_audio(args.wav)
    elif args.record:
        audio = record_audio(args.record)
    else:
        parser.error("Provide --wav, --record, or --merge")
        return

    from unmute.speaker_gate import SpeakerGate
    gate = SpeakerGate.__new__(SpeakerGate)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    gate._session = ort.InferenceSession(args.model, sess_options=so)

    emb = gate.extract_embedding(audio, 16000)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    print(f"Enrollment saved to {out_path}")
    print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
