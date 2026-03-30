# EchoTrack

**Real-time Deepfake & Synthetic Speech Trajectory Tracker** built with JAX.

EchoTrack listens to live audio and continuously tracks the probability that the speech is synthesized or cloned, building a **temporal trajectory** of deepfake likelihood + acoustic artifact heatmap.

### Key Features
- **Week 1**: Streaming audio pipeline + microphone demo
- **Week 2**: JAX-native Mamba-2 baseline + training script (ASVspoof 5)
- **Week 3–4**: Stateful streaming inference (`jax.lax.scan` + cache) + bidirectional fusion + custom anomaly features

### Installation
```bash
uv venv
uv pip install -e ".[dev]"
```

### Quick Start
```bash
# Train baseline
python scripts/train_baseline.py

# Live demo (after training)
python -m echotrack.demo.microphone_demo
```

### Project Structure

```
EchoTrack/
├── src/echotrack/
│   ├── __init__.py
│   ├── data/
│   │   ├── streaming_buffer.py
│   │   └── asvspoof_loader.py
│   ├── utils/
│   │   └── audio_utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mamba_jax.py
│   └── demo/
│       └── microphone_demo.py
├── scripts/
│   └── train_baseline.py
├── notebooks/
├── configs/
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

### Week 2 — Baseline Model Completed
- Improved JAX-native Mamba-2 SSM
- Simple classification head
- Training script on ASVspoof 5 subset

**Next (Weeks 3–4)**: Stateful inference with `jax.lax.scan`, bidirectional fusion, custom phase/phoneme features, latency < 300 ms.

### Dataset
ASVspoof 5 (jungjee/asvspoof5)

Made with ❤️ for privacy-first real-time deepfake defense.
