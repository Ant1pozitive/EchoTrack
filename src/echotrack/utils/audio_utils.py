"""
Utility functions for audio preprocessing in JAX.
"""

import jax.numpy as jnp


def normalize_audio(audio: jnp.ndarray) -> jnp.ndarray:
    """Normalize audio to zero mean and unit variance (or [-1,1])."""
    return audio / (jnp.max(jnp.abs(audio)) + 1e-8)


def resample_if_needed(audio: jnp.ndarray, orig_sr: int, target_sr: int = 16000) -> jnp.ndarray:
    """Placeholder for resampling. In Week 1 we assume 16kHz input."""
    if orig_sr == target_sr:
        return audio
    # TODO: implement proper resampling with scipy or jax-friendly method in future weeks
    print(f"Warning: resampling from {orig_sr} to {target_sr} not implemented yet.")
    return audio
