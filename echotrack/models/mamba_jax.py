"""
JAX-native Mamba-2 style SSM for streaming deepfake detection.
Minimal, efficient implementation using jax.lax.scan.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Tuple, Optional


class Mamba2Block(nn.Module):
    """Simplified Mamba-2 block with selective SSM."""
    d_model: int = 64          # small for Week 2 baseline
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int = 8
    bias: bool = False
    conv_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, cache: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: [batch, length, d_model]  (for streaming length=1)
        Returns: output, new_cache
        """
        batch, length, dim = x.shape

        # Input projection
        xz = nn.Dense(features=dim * 2, use_bias=self.bias)(x)
        x, z = jnp.split(xz, 2, axis=-1)

        # Conv1d (depthwise)
        conv = nn.Conv(features=dim, kernel_size=(self.d_conv,), padding="CAUSAL", feature_group_count=dim)(x)
        x = nn.silu(conv)

        # Selective SSM parameters
        dt = nn.Dense(features=self.dt_rank)(x)
        B = nn.Dense(features=self.d_state)(x)
        C = nn.Dense(features=self.d_state)(x)

        # Delta (softplus)
        dt = jax.nn.softplus(dt + jnp.log(jnp.exp(0.5) - 1))  # stable softplus init

        # SSM state update (simple selective scan)
        if cache is None:
            cache = jnp.zeros((batch, self.d_state, dim))

        # For streaming we use lax.scan internally in inference, here simplified recurrent step
        # (full scan version will be in Week 3 for bidirectional)
        y, new_cache = self._selective_scan(x, dt, B, C, cache)

        # Output projection + gating
        y = y * nn.silu(z)
        out = nn.Dense(features=dim, use_bias=self.bias)(y)

        return out, new_cache

    def _selective_scan(self, x: jnp.ndarray, dt: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cache: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Minimal selective scan step (recurrent for streaming)."""
        # This is a simplified version; in real Mamba-2 it's more optimized with parallel scan
        # For Week 2 we keep it causal and recurrent
        A = -jnp.exp(jnp.arange(self.d_state))  # fixed A for simplicity
        dt = dt[..., None, :]  # broadcast
        x = x[..., None, :] * dt
        B = B[..., None, :]

        # State update
        cache = cache * jnp.exp(A * dt) + x * B
        y = jnp.einsum("b s d, b l s -> b l d", cache, C)

        return y, cache


class Mamba2Classifier(nn.Module):
    """Full baseline model: Mamba-2 + head for binary deepfake classification."""
    d_model: int = 64
    num_classes: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True, cache: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: [batch, length, 1] raw waveform chunk
        x = x[..., None]  # add channel dim if needed

        # Simple frontend (can be replaced with XLSR later)
        x = nn.Conv(features=self.d_model, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)

        # Mamba block
        x, new_cache = Mamba2Block(d_model=self.d_model)(x, cache)

        # Global average pooling for classification
        x = jnp.mean(x, axis=1)  # [batch, d_model]

        # Head
        logits = nn.Dense(features=self.num_classes)(x)
        probs = nn.softmax(logits, axis=-1)

        return probs, new_cache
