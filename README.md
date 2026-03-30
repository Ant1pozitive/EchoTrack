# EchoTrack

**Real-time Deepfake & Synthetic Speech Trajectory Tracker** built with JAX.

EchoTrack listens to live audio and continuously tracks the probability that the speech is synthesized or cloned, building a **temporal trajectory** of deepfake likelihood while highlighting acoustic artifacts over time.

### Key Features

- **Week 1**: Streaming audio pipeline, microphone demo, ASVspoof 5 loader
- **Week 2**: JAX-native Mamba-2 baseline model + classification head + training script
- Fully local, low-latency, designed for streaming inference

### Installation

```bash
uv venv
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"
```

### Quick Start

```bash
# Microphone demo with baseline model
python -m echotrack.demo.microphone_demo

# Train baseline model (uses small subset for speed)
python scripts/train_baseline.py
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

- Improved JAX-native Mamba-2 style SSM (unidirectional, selective mechanism, better stability)
- Simple classification head (bonafide vs spoof)
- Training script on ASVspoof 5 (small subset for fast iteration)
- Model can be trained to reasonable baseline (target EER < 8% on dev set with full training)

**Next (Week 3)**: Stateful streaming inference with `jax.lax.scan`, bidirectional fusion, full training loop + EER calculation.

### Dataset

ASVspoof 5 from Hugging Face: `jungjee/asvspoof5`

### License
MIT

Made with ❤️ for privacy-first real-time deepfake defense.
