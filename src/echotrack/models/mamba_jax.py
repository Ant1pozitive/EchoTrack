"""
Improved JAX-native Mamba-2 style SSM for streaming deepfake detection.
Better stability, proper selective parameters, and cleaner recurrent step.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional

class Mamba2Block(nn.Module):
    """Improved Mamba-2 block with selective SSM."""
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int = 8
    bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, cache: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch, length, dim = x.shape

        # Input projection + gating
        xz = nn.Dense(features=dim * 2, use_bias=self.bias)(x)
        x, z = jnp.split(xz, 2, axis=-1)

        # Depthwise convolution (causal)
        x = nn.Conv(
            features=dim,
            kernel_size=(self.d_conv,),
            padding="CAUSAL",
            feature_group_count=dim,
            use_bias=True
        )(x)
        x = nn.silu(x)

        # Selective parameters
        dt = nn.Dense(features=self.dt_rank)(x)
        B = nn.Dense(features=self.d_state)(x)
        C = nn.Dense(features=self.d_state)(x)

        # Stable delta
        dt = jax.nn.softplus(dt)

        # SSM state
        if cache is None:
            cache = jnp.zeros((batch, self.d_state, dim), dtype=jnp.float32)

        # Simplified but stable selective scan (recurrent for streaming)
        y, new_cache = self._selective_scan(x, dt, B, C, cache)

        # Output gating
        y = y * nn.silu(z)
        out = nn.Dense(features=dim, use_bias=self.bias)(y)

        return out, new_cache

    def _selective_scan(self, x: jnp.ndarray, dt: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cache: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Stable selective scan step."""
        # Fixed A matrix (diagonal decay)
        A = -jnp.exp(jnp.arange(self.d_state, dtype=jnp.float32))

        # Broadcast
        dt = dt[..., None, :]                    # [batch, length, 1, dt_rank] -> adjust as needed
        B = B[..., None, :]                      # [batch, length, 1, d_state]
        x = x[..., None, :] * dt                 # element-wise modulation

        # Recurrent state update
        cache = cache * jnp.exp(A * dt.mean(axis=-1, keepdims=True)) + x * B
        y = jnp.einsum("b s d, b l s -> b l d", cache, C)

        return y, cache

class Mamba2Classifier(nn.Module):
    """Baseline classifier: Mamba-2 block + simple head."""
    d_model: int = 64
    num_classes: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True, cache: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x shape: [batch, length] or [batch, length, 1]
        if x.ndim == 2:
            x = x[..., None]

        # Lightweight frontend
        x = nn.Conv(features=self.d_model, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)

        # Mamba block
        x, new_cache = Mamba2Block(d_model=self.d_model)(x, cache)

        # Pooling + head
        x = jnp.mean(x, axis=1)  # global average
        logits = nn.Dense(features=self.num_classes)(x)
        probs = nn.softmax(logits, axis=-1)

        return probs, new_cache