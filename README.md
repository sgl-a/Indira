# Indira

[![tests](https://github.com/sgl-a/Indira/actions/workflows/tests.yml/badge.svg)](https://github.com/sgl-a/Indira/actions/workflows/tests.yml)

**An artificial being who ages from 10 to 70 over 72 hours, in conversation with her mother.**

Indira is a fully-local AI performance system built for a durational theatrical installation. Across a continuous 72-hour run she grows from a ten-year-old into a woman of seventy — speaking, remembering, and changing in real time alongside a live actress playing her mother. Everything runs on-device: no cloud calls, no network dependency, nothing that can fail because a venue's wifi dropped.

Under the hood it is a modular pipeline — speech-to-text, an LLM brain, text-to-speech, an eight-stage aging engine, and a two-tier memory that consolidates conversations into first-person recollections as the details fade. Every component is swappable through configuration rather than code.

> **Status:** The conversational actor works today — voice in, voice out, with memory and real-time aging. Computer vision and the projected avatar are designed but not yet built. See [ARCHITECTURE.md](ARCHITECTURE.md) for implemented-versus-planned status on every component.

## Quick Start

### 1. Prerequisites

| Requirement | Install |
|-------------|---------|
| **Python 3.13** | `brew install python@3.13` |
| **Ollama** | `brew install ollama` |
| **Homebrew** (if missing) | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |

### 2. Install Dependencies

```bash
cd indira

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate

# Install with the providers the default config uses
pip install -e ".[whisper,tts,memory,dev]"

# Extras: whisper (mlx-whisper STT) · tts (mlx-audio, runs Qwen3-TTS)
#         memory (ChromaDB) · tools (Gradio memory explorer) · dev (pytest)
```

> **Note:** `pip install -e .` alone installs only the core runtime — no STT, no TTS, no vector memory. The default configuration needs the `whisper`, `tts`, and `memory` extras to start.

### 3. Start Ollama

```bash
# Run in a separate terminal tab
#ollama serve

./scripts/start_ollama.sh 
```

> **Tip:** Always use `scripts/start_ollama.sh` (not a bare `ollama serve`) — it sets `OLLAMA_NUM_PARALLEL=2` so background memory-consolidation calls don't evict the conversation's KV cache. Model keep-alive is handled per-request by the app (`llm.keep_alive: -1` — the model stays loaded for the whole run and is explicitly unloaded on `/quit`).

### 4. Pull a Model

```bash
ollama pull <modelname>
```

> **Note:** Model download is a one-time operation. After pulling, everything runs 100% locally — no internet required.

### 5. Run Indira

```bash
source .venv/bin/activate
python3 -m src.main
```

## Launch Options

```bash
python3 -m src.main [flags]
```

| Flag | Description |
|------|-------------|
| *(none)* | Text mode. Auto-resumes a running performance (hour, age, conversation transcript) if still inside the 72h window |
| `--mode audio` | Voice mode: microphone → Whisper STT → LLM → TTS → speaker (VAD-gated, mic pauses while she speaks) |
| `--mode text` | Text mode, explicit (the default) |
| `--fresh` | Start the performance from hour 0 — discards the saved performance clock **and** the conversation transcript |
| `--env NAME` | Overlay `config/NAME.yaml` on top of `default.yaml` (e.g. `--env development`) |
| `--config DIR` | Config directory (default: `config/`) |
| `--log-level LEVEL` | Override log level (`DEBUG` shows consolidation, TTS instructs, cache prewarms) |
| `--list-providers` | Print registered STT/LLM/TTS/Memory providers and exit |

Common recipes:

```bash
# Normal session (resumes mid-performance after a crash/restart)
python3 -m src.main

# Voice mode
python3 -m src.main --mode audio

# Dev session: tiny memory window so consolidation fires after ~5 exchanges
# (production waits ~40 — see memory.history_max_turns)
python3 -m src.main --env development

# Fresh start with full internals visible
python3 -m src.main --fresh --log-level DEBUG

# One-off config overrides via env vars (double underscore = section nesting)
INDIRA__LLM__MODEL=qwen3.5:9b python3 -m src.main
INDIRA__MEMORY__CONSOLIDATION__ENABLED=false python3 -m src.main
INDIRA_ENV=development python3 -m src.main            # same as --env development
```

Override precedence: `config/default.yaml` → `config/<env>.yaml` → `INDIRA__*` env vars.

## Text Mode Commands

| Command | Description |
|---------|-------------|
| `/status` | Show age, emotion, memory count, conversation turns |
| `/age N` | Set character to N years old (e.g., `/age 25` to jump to the 25-30 age stage) |
| `/memory` | View recent stored memories |
| `/lobotomy` | Erase all stored memories |
| `/help` | Show all commands |
| `/quit` | Exit gracefully |

## Swapping Models

### Via Config

Edit `config/default.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "qwen3.5:27b"  # ← change to any Ollama model
```

### Via Environment Variable

```bash
INDIRA__LLM__MODEL=qwen3:30b-a3b python3 -m src.main
```

### Available Providers

```bash
python3 -m src.main --list-providers
```

```
STT: whisper
LLM: ollama
TTS: system, qwen
MEMORY: chroma, simple
```

## Benchmarking

Compare models for speed and character consistency:

```bash
# Compare LLM models
python3 -m scripts.benchmark llm --models llama3.1:8b,qwen3.5:27b

# Compare TTS providers
python3 -m scripts.benchmark tts --providers system

# Run all benchmarks
python3 -m scripts.benchmark full
```

## Database Explorer

Explore, search, and manage memories stored in ChromaDB (Web UI or CLI mode):

```bash
# Run the Interactive Web Dashboard (Gradio)
python3 scripts/chroma_explorer.py

# Run in command-line mode (Rich-formatted table)
python3 scripts/chroma_explorer.py --cli

# Specify a custom database directory, collection, or port
python3 scripts/chroma_explorer.py --dir data/memory --collection indira_memories --port 7860
```

The Web UI is available at `http://127.0.0.1:7860/` by default and allows you to:
- Browse all stored memories and filter by age stage or memory type.
- Perform semantic similarity searches with vector distance metrics.
- Export/Backup all memories to a JSON file inside `data/memory/`.
- Add or delete memories for debugging purposes.

## Project Structure

```
indira/
├── config/default.yaml           ← All model & behavior configuration
├── src/
│   ├── core/
│   │   ├── interfaces/           ← Abstract provider interfaces (STT, LLM, TTS, Memory)
│   │   ├── orchestrator.py       ← Main pipeline: STT → Memory → LLM → TTS
│   │   ├── state.py              ← Central state (age, emotion, conversation history)
│   │   ├── age_engine.py         ← 8-stage age progression with personality prompts
│   │   ├── config.py             ← YAML + env var config loader
│   │   └── registry.py           ← Provider registry (config-driven, lazy imports)
│   ├── providers/
│   │   ├── llm/ollama_provider   ← Ollama API with emotion tag parsing
│   │   ├── stt/whisper_provider  ← mlx-whisper (Apple Silicon, Spanish)
│   │   ├── tts/qwen_tts_provider ← Qwen3-TTS via MLX (default)
│   │   ├── tts/system_provider   ← macOS 'say' fallback
│   │   └── memory/               ← simple_provider (keyword) + chroma_provider (semantic)
│   └── main.py                   ← CLI entry point
├── profiles/                     ← 8 age personality profiles (10-15 through 60-70)
├── data/memory/                  ← ChromaDB persistent storage (auto-created)
├── scripts/
│   ├── benchmark.py              ← Model comparison tool
│   └── chroma_explorer.py        ← Database explorer (Web UI / CLI)
└── pyproject.toml
```

## Ollama Management

```bash
# Start / Stop
ollama serve                   # Start in current terminal (Ctrl+C to stop) (see below)
./scripts/start_ollama.sh      # recommended to run with this instead
pkill ollama                   # Stop from any terminal

# Check if running
pgrep -fl ollama               # Shows process if running
curl -s http://localhost:11434  # Returns "Ollama is running" if active

# Model management
ollama list                    # Show downloaded models
ollama pull <model>            # Download a model
ollama rm <model>              # Remove a model (frees disk space)
```

> **RAM note:** Ollama loads model weights into memory (~5-20GB depending on model). While Indira runs, the LLM stays pinned (`llm.keep_alive: -1` — no cold starts after silences) and is unloaded on `/quit`; the embedding model self-unloads after ~5 min. Use `ollama ps` to see what's loaded, `ollama stop <model>` to free RAM manually after a crash.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design, data flow, and implemented-vs-planned status
- [TECH_STACK.md](TECH_STACK.md) — Technology choices, models under evaluation, and alternatives

## License

Not licensed yet — all rights reserved for the moment. The work is in progress and
the piece it belongs to has not premiered, so I'd rather choose a license once both
are settled than pick one I can't walk back.

The code is public to be read, and I'm glad for it to be useful that way. If you want
to run it, build on it, or use any part of it before a license lands, open an issue and
ask — the answer is likely yes.

Note that this repository ships **no model weights**. Ollama models, `mlx-whisper`, and
Qwen3-TTS are pulled at setup and carry their own licenses; Gemma in particular is
distributed under the Gemma Terms of Use rather than an OSI-approved license.
