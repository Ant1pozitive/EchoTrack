"""
Week 2 Training script for EchoTrack baseline model.
Trains Mamba2Classifier on ASVspoof 5 (small subset for fast iteration).
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn
from tqdm import tqdm
import numpy as np

from echotrack.models.mamba_jax import Mamba2Classifier
from echotrack.data.asvspoof_loader import load_asvspoof5

def collate_batch(batch):
    """Simple collate for raw waveforms and labels."""
    waveforms = [jnp.array(item["audio"][:3200]) for item in batch]  # truncate/pad to fixed length for simplicity
    labels = jnp.array([item["label"] for item in batch])
    # Pad/truncate to same length
    max_len = max(w.shape[0] for w in waveforms)
    padded = [jnp.pad(w, (0, max_len - w.shape[0]))[:3200] for w in waveforms]
    return jnp.stack(padded), labels

def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 5
    learning_rate = 1e-3
    key = jax.random.PRNGKey(42)

    # Model
    model = Mamba2Classifier(d_model=64)
    dummy_input = jnp.zeros((1, 3200))
    variables = model.init(key, dummy_input)
    params = variables['params']

    # Optimizer
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Loss
    def loss_fn(params, batch, labels):
        probs, _ = model.apply({'params': params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(probs, labels).mean()
        return loss

    @jax.jit
    def train_step(state, batch, labels):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, labels)
        state = state.apply_gradients(grads=grads)
        return state, loss

    print("Starting baseline training on ASVspoof 5 (small subset)...")

    # Load small subset (streaming + take)
    train_data = list(load_asvspoof5(split="train", streaming=True))[:200]  # small for Week 2

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch_data = train_data[i:i+batch_size]
            waveforms, labels = collate_batch(batch_data)
            state, loss = train_step(state, waveforms, labels)
            epoch_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {epoch_loss / (len(train_data)//batch_size + 1):.4f}")

    # Save params (simple)
    jnp.save("checkpoints/baseline_params.npy", state.params)
    print("Training finished. Model saved to checkpoints/baseline_params.npy")

    # TODO Week 3: add EER evaluation on dev set

if __name__ == "__main__":
    main()