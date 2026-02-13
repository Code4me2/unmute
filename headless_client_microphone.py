#!/usr/bin/env python3
"""
Real-time microphone client for Unmute - stream audio from Mac to remote backend.

This client:
- Records audio from your microphone (Mac/Linux/Windows)
- Streams it to Unmute backend via WebSocket (can be over Tailscale)
- Receives and plays TTS responses in real-time
- Works completely over the network - no local GPU needed

Usage:
    # Connect to local backend
    python headless_client_microphone.py

    # Connect to remote backend via Tailscale
    python headless_client_microphone.py --server-url ws://100.64.1.1:8000

    # List available audio devices
    python headless_client_microphone.py --list-devices
"""

import argparse
import asyncio
import base64
import logging
import queue
import sys
import threading
from typing import Optional

try:
    import numpy as np
    import pyaudio
    import sphn
    import websockets
    from pydantic import Field, TypeAdapter
    from typing_extensions import Annotated
except ImportError as e:
    print(f"Error: Required dependencies not installed: {e}")
    print("\nPlease run:")
    print("  pip install websockets numpy sphn pydantic pyaudio")
    print("\nOn macOS, you may also need:")
    print("  brew install portaudio")
    sys.exit(1)

# Import from unmute package
try:
    import unmute.openai_realtime_api_events as ora
    from unmute.kyutai_constants import SAMPLE_RATE
    from unmute.llm.system_prompt import SmalltalkInstructions
except ImportError:
    print("Error: Could not import unmute package.")
    print("Make sure you're running this from the unmute directory,")
    print("or install the unmute package.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("mic_client")


class MicrophoneClient:
    """Real-time microphone client for Unmute."""

    def __init__(
        self,
        server_url: str = "ws://localhost:8000",
        voice: Optional[str] = None,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        self.server_url = server_url
        self.voice = voice or "default"
        self.input_device = input_device
        self.output_device = output_device

        self.websocket: Optional[websockets.ClientConnection] = None
        self.audio_interface = pyaudio.PyAudio()
        self.opus_reader = None
        self.opus_writer = None

        # Queues for audio streaming
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Control flags
        self.recording = False
        self.playing = False
        self.running = True

    def list_audio_devices(self):
        """List all available audio devices."""
        print("\n=== Available Audio Devices ===\n")

        print("INPUT DEVICES (Microphones):")
        for i in range(self.audio_interface.get_device_count()):
            info = self.audio_interface.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                print(f"      Channels: {info['maxInputChannels']}, "
                      f"Sample Rate: {int(info['defaultSampleRate'])} Hz")

        print("\nOUTPUT DEVICES (Speakers):")
        for i in range(self.audio_interface.get_device_count()):
            info = self.audio_interface.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                print(f"      Channels: {info['maxOutputChannels']}, "
                      f"Sample Rate: {int(info['defaultSampleRate'])} Hz")
        print()

    def _input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Input status: {status}")

        # Convert bytes to numpy array
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.input_queue.put(audio)

        return (None, pyaudio.paContinue)

    def _output_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio output stream."""
        if status:
            logger.warning(f"Output status: {status}")

        try:
            # Get audio from queue
            audio = self.output_queue.get_nowait()

            # Pad or trim to exact frame count
            if len(audio) < frame_count:
                audio = np.pad(audio, (0, frame_count - len(audio)))
            elif len(audio) > frame_count:
                audio = audio[:frame_count]

            # Convert to int16 bytes
            audio_bytes = (audio * 32768.0).astype(np.int16).tobytes()
            return (audio_bytes, pyaudio.paContinue)

        except queue.Empty:
            # No audio available, return silence
            silence = np.zeros(frame_count, dtype=np.int16).tobytes()
            return (silence, pyaudio.paContinue)

    async def connect(self):
        """Establish WebSocket connection to backend."""
        websocket_url = f"{self.server_url.strip('/')}/v1/realtime"
        logger.info(f"Connecting to {websocket_url}")

        try:
            self.websocket = await websockets.connect(
                websocket_url,
                subprotocols=[websockets.Subprotocol("realtime")],
            )
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            logger.error(f"Make sure the backend is running and accessible at {websocket_url}")
            raise

        # Initialize opus codec
        self.opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self.opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        # Send initial session update
        await self._send_session_update()

        logger.info("✓ Connected successfully!")

    async def _send_session_update(self):
        """Send session configuration to backend."""
        session_update = ora.SessionUpdate(
            session=ora.SessionConfig(
                instructions=SmalltalkInstructions(),
                voice=self.voice,
                allow_recording=False,
            )
        )
        await self.websocket.send(session_update.model_dump_json())

    async def send_audio_loop(self):
        """Send audio from microphone to backend."""
        CHUNK_SIZE = 1920  # 40ms at 48kHz

        logger.info("🎤 Started audio input")

        while self.running:
            try:
                # Get audio from input queue
                audio = self.input_queue.get(timeout=0.1)

                # Encode to opus
                opus_bytes = self.opus_writer.append_pcm(audio)
                if opus_bytes:
                    # Encode to base64 and send
                    encoded = base64.b64encode(opus_bytes).decode("utf-8")
                    message = ora.InputAudioBufferAppend(audio=encoded)
                    await self.websocket.send(message.model_dump_json())

            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Error sending audio: {e}")
                break

    async def receive_audio_loop(self):
        """Receive audio from backend and play it."""
        logger.info("🔊 Started audio output")
        current_audio_chunks = []

        while self.running:
            try:
                message_raw = await asyncio.wait_for(
                    self.websocket.recv(), timeout=0.1
                )

                message: ora.ServerEvent = TypeAdapter(
                    Annotated[ora.ServerEvent, Field(discriminator="type")]
                ).validate_json(message_raw)

                if isinstance(message, ora.ResponseCreated):
                    logger.info("🤖 Response started")
                    current_audio_chunks = []

                elif isinstance(message, ora.ResponseTextDelta):
                    # Print text as it arrives
                    print(message.delta, end="", flush=True)

                elif isinstance(message, ora.ResponseTextDone):
                    print()  # New line after text

                elif isinstance(message, ora.ResponseAudioDelta):
                    # Decode and queue audio for playback
                    binary_audio = base64.b64decode(message.delta)
                    pcm = self.opus_reader.append_bytes(binary_audio)

                    if pcm.size:
                        # Queue audio chunks for playback
                        # Split into smaller chunks for smoother playback
                        chunk_size = 1920
                        for i in range(0, len(pcm), chunk_size):
                            chunk = pcm[i:i + chunk_size]
                            self.output_queue.put(chunk)

                elif isinstance(message, ora.ResponseAudioDone):
                    logger.info("✓ Response complete")

                elif isinstance(message, ora.Error):
                    logger.error(f"Server error: {message.error.message}")

                elif isinstance(message, ora.UnmuteToolCallEvent):
                    logger.info(
                        "🔧 Tool call: %s(%s)",
                        message.tool_name,
                        message.arguments[:80],
                    )

                elif isinstance(message, ora.InputAudioBufferSpeechStarted):
                    logger.info("👂 Listening...")

                elif isinstance(message, ora.InputAudioBufferSpeechStopped):
                    logger.info("⏸️  Speech stopped, processing...")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Error receiving audio: {e}")
                break

    def start_audio_streams(self):
        """Start PyAudio input and output streams."""
        logger.info("Starting audio streams...")

        # Input stream (microphone)
        self.input_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=1920,  # 40ms chunks
            stream_callback=self._input_callback,
        )

        # Output stream (speakers)
        self.output_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=1920,
            stream_callback=self._output_callback,
        )

        self.input_stream.start_stream()
        self.output_stream.start_stream()

        logger.info("✓ Audio streams started")

    def stop_audio_streams(self):
        """Stop PyAudio streams."""
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()

        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()

    async def run(self):
        """Main run loop."""
        try:
            await self.connect()

            # Start audio streams
            self.start_audio_streams()

            logger.info("\n" + "="*50)
            logger.info("🎙️  Microphone client running!")
            logger.info("Speak into your microphone to interact")
            logger.info("Press Ctrl+C to stop")
            logger.info("="*50 + "\n")

            # Run send and receive loops concurrently
            await asyncio.gather(
                self.send_audio_loop(),
                self.receive_audio_loop(),
            )

        except KeyboardInterrupt:
            logger.info("\n⏹️  Stopping...")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self.running = False
            self.stop_audio_streams()

            if self.websocket:
                await self.websocket.close()

            self.audio_interface.terminate()
            logger.info("✓ Disconnected")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time microphone client for Unmute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--server-url",
        default="ws://localhost:8000",
        help="WebSocket URL of Unmute backend (default: ws://localhost:8000)\n"
             "For Tailscale: ws://100.64.1.1:8000",
    )

    parser.add_argument(
        "--voice",
        type=str,
        help="Voice to use for TTS",
    )

    parser.add_argument(
        "--input-device",
        type=int,
        help="Audio input device index (use --list-devices to see options)",
    )

    parser.add_argument(
        "--output-device",
        type=int,
        help="Audio output device index (use --list-devices to see options)",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )

    args = parser.parse_args()

    client = MicrophoneClient(
        server_url=args.server_url,
        voice=args.voice,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    if args.list_devices:
        client.list_audio_devices()
        return

    asyncio.run(client.run())


if __name__ == "__main__":
    main()
