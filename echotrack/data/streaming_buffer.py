"""
Streaming audio buffer for real-time processing.
Handles chunked audio input with overlap and converts to JAX arrays efficiently.
"""

import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import Optional, Tuple


class AudioStreamingBuffer:
    """
    Stateful buffer for streaming audio at 16kHz.
    Supports overlapping chunks for smooth feature extraction.
    """

    def __init__(self, chunk_size: int = 320, overlap: int = 160, sample_rate: int = 16000):
        """
        Args:
            chunk_size: Number of samples per processing chunk (e.g. 320 samples = 20ms at 16kHz)
            overlap: Overlap between consecutive chunks in samples
            sample_rate: Audio sample rate in Hz
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.buffer: deque = deque(maxlen=chunk_size + overlap)
        self.step_size = chunk_size - overlap

    def add_samples(self, samples: np.ndarray) -> None:
        """Add new raw audio samples to the buffer."""
        for sample in samples:
            self.buffer.append(sample)

    def get_next_chunk(self) -> Optional[np.ndarray]:
        """
        Return the next full chunk if available.
        Returns None if not enough samples yet.
        """
        if len(self.buffer) < self.chunk_size:
            return None
        chunk = np.array(list(self.buffer)[:self.chunk_size], dtype=np.float32)
        # Remove the non-overlapping part
        for _ in range(self.step_size):
            self.buffer.popleft()
        return chunk

    def to_jax(self, chunk: np.ndarray) -> jnp.ndarray:
        """Convert numpy chunk to JAX array (float32 normalized)."""
        # Normalize to [-1, 1] range if needed (sounddevice usually gives float32 already)
        jax_chunk = jnp.asarray(chunk, dtype=jnp.float32)
        if jnp.max(jnp.abs(jax_chunk)) > 1.0:
            jax_chunk = jax_chunk / 32768.0  # assume int16 otherwise
        return jax_chunk
