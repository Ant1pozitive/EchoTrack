# EchoTrack

**Real-time Deepfake & Synthetic Speech Trajectory Tracker** built with JAX.

EchoTrack listens to live audio (microphone or WebRTC) and continuously tracks the probability that the speech is synthesized or cloned. Instead of a simple binary classifier, it builds a **temporal trajectory** of deepfake likelihood while highlighting acoustic artifacts (phase anomalies, unnatural phoneme transitions, spectral inconsistencies) over time.

### Key Features (Week 1 MVP)
- Streaming audio input from microphone (16 kHz, chunked)
- JAX-native data pipeline with efficient tensor conversion
- Dataset loader for ASVspoof 5 (the latest 2025/2026 benchmark)
- Basic streaming buffer and preprocessing utilities
- Ready for Mamba-2 / SSM core in the next weeks

### Why JAX?
Pure JAX + Flax/NNX allows custom stateful streaming with `jax.lax.scan`, bidirectional SSMs, and excellent on-device performance.

### Installation

```bash
# Recommended: use uv or poetry
uv venv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Quick Start (Microphone Demo)

```bash
python -m echotrack.demo.microphone_demo
```

This will open a live audio stream, process chunks, and print basic statistics.

### Project Structure (Week 1)

```
echotrack/
├── src/echotrack/
│   ├── data/           # streaming loaders and buffers
│   ├── utils/          # jax helpers, audio utils
│   └── demo/           # quick microphone demo
├── notebooks/
├── configs/
├── scripts/
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

### Roadmap

See the full roadmap in the repository wiki or future updates.  
**Week 1** — Setup, streaming audio pipeline, ASVspoof 5 loader.

### Dataset

We use [jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) on Hugging Face.

### License
MIT

Made with ❤️ for privacy-first real-time deepfake defense.
