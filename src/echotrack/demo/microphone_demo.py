"""
Week 2 Microphone Demo: Baseline Mamba-2 model in action.
"""

import sounddevice as sd
import numpy as np
import jax
import jax.numpy as jnp
import time
from echotrack.data.streaming_buffer import AudioStreamingBuffer
from echotrack.models.mamba_jax import Mamba2Classifier


# Initialize model (random weights for Week 2 demo)
model = Mamba2Classifier(d_model=64)
key = jax.random.PRNGKey(42)
dummy_input = jnp.zeros((1, 320, 1))  # batch, length, 1
params = model.init(key, dummy_input)[ 'params']


def microphone_demo(duration: float = 60.0):
    sample_rate = 16000
    chunk_size = 320
    overlap = 160

    buffer = AudioStreamingBuffer(chunk_size=chunk_size, overlap=overlap, sample_rate=sample_rate)

    print("🎤 EchoTrack Week 2 Demo (Mamba-2 Baseline)")
    print("Model loaded. Listening for deepfake probability...\n")

    def callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(status)

        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        buffer.add_samples(mono.astype(np.float32))

        chunk_np = buffer.get_next_chunk()
        if chunk_np is not None:
            jax_chunk = buffer.to_jax(chunk_np)
            jax_chunk = jnp.expand_dims(jax_chunk, axis=0)  # [1, length]

            # Inference
            probs, _ = model.apply({'params': params}, jax_chunk)
            deepfake_prob = float(probs[0, 1])  # spoof class

            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Chunk | Deepfake prob: {deepfake_prob:.4f} | {'🟥 SPOOF' if deepfake_prob > 0.6 else '🟩 REAL'}")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                            blocksize=chunk_size, callback=callback):
            sd.sleep(int(duration * 1000))
    except KeyboardInterrupt:
        print("\nDemo stopped.")


if __name__ == "__main__":
    microphone_demo()
