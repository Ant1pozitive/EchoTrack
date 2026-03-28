"""
Simple real-time microphone demo for EchoTrack Week 1.
Records audio chunks and prints basic statistics.
"""

import sounddevice as sd
import numpy as np
import time
from echotrack.data.streaming_buffer import AudioStreamingBuffer


def microphone_demo(duration: float = 30.0, chunk_duration_ms: int = 20):
    """
    Run a live microphone demo.

    Args:
        duration: How long to record in seconds
        chunk_duration_ms: Processing chunk size in milliseconds
    """
    sample_rate = 16000
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)   # e.g. 320 samples
    overlap = chunk_size // 2

    buffer = AudioStreamingBuffer(chunk_size=chunk_size, overlap=overlap, sample_rate=sample_rate)

    print("🎤 EchoTrack Microphone Demo started")
    print(f"Chunk size: {chunk_size} samples ({chunk_duration_ms} ms)")
    print("Press Ctrl+C to stop\n")

    def callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Status: {status}")

        # indata shape: (frames, channels) — take mono
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        buffer.add_samples(mono.astype(np.float32))

        chunk = buffer.get_next_chunk()
        if chunk is not None:
            jax_chunk = buffer.to_jax(chunk)
            # Week 1: simple statistics only
            rms = float(jnp.sqrt(jnp.mean(jax_chunk ** 2)))
            peak = float(jnp.max(jnp.abs(jax_chunk)))
            print(f"[{time.strftime('%H:%M:%S')}] Chunk processed | RMS: {rms:.4f} | Peak: {peak:.4f} | Samples: {len(chunk)}")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                            blocksize=chunk_size, callback=callback):
            print(f"Listening for {duration} seconds...")
            sd.sleep(int(duration * 1000))
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    microphone_demo(duration=60.0)
