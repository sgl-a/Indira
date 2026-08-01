# 🎭 AI Actor Installation

A multimodal AI theatrical performance system for a 72-hour durational installation where an artificial "being" ages from 10 to 70 in real-time, interacting with a live actress.

## Overview

The AI entity can act and interact with a live actress over 72 hours. It ages in real-time, evolving from childhood (10) to old age (70), while maintaining character, conveying emotion, and performing as a theatrical actor.

> **Status (2026-06-30):** Working today = an **audio/text conversational actor** (STT → LLM → TTS, memory, aging personality). Vision (cameras), the visual **avatar/projection**, and proactive behavior are **designed but not yet built** — the year-end 2026 target is the full system. See status tags in [ARCHITECTURE.md](ARCHITECTURE.md) and [TODO.md](TODO.md).

### Architecture

The system follows a **provider pattern** — every component is swappable via configuration:

```
Microphone → [STT Provider] → [LLM Provider] → [TTS Provider] → Speaker
                                     ↕
                              [Memory Provider]
                              [Age Engine]
```

## Quick Start

### 1. Prerequisites

| Requirement | Install |
|-------------|---------|
| **Python 3.13** | `brew install python@3.13` |
| **Ollama** | `brew install ollama` |
| **Homebrew** (if missing) | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |

### 2. Install Dependencies

```bash
cd ai-actor-project

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate

# Install the project
pip install -e .

# (Optional) Install with extra providers
pip install -e ".[whisper,tts,dev]"
```

### 3. Start Ollama

```bash
# Run in a separate terminal tab
ollama serve
```

> **Tip:** Use `ollama serve` instead of `brew services start ollama` for manual control — just `Ctrl+C` to stop. Either way, Ollama auto-unloads models from RAM after 5 minutes of inactivity.

### 4. Pull a Model

```bash
# Production brain (17GB, 262K context, multimodal)
ollama pull qwen3.5:27b

# Fast dev/testing model (18GB, MoE — much faster responses)
ollama pull qwen3:30b-a3b

# Lightweight fallback (~5GB, good for quick iteration)
ollama pull llama3.1:8b
```

> **Note:** Model download is a one-time operation. After pulling, everything runs 100% locally — no internet required.

### 5. Run the AI Actor

```bash
source .venv/bin/activate
python3 -m src.main
```

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
AI_ACTOR__LLM__MODEL=qwen3:30b-a3b python3 -m src.main
```

### Available Providers

```bash
python3 -m src.main --list-providers
```

```
STT: whisper
LLM: ollama
TTS: kokoro, system
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
python3 scripts/chroma_explorer.py --dir data/memory --collection ai_actor_memories --port 7860
```

The Web UI is available at `http://127.0.0.1:7860/` by default and allows you to:
- Browse all stored memories and filter by age stage or memory type.
- Perform semantic similarity searches with vector distance metrics.
- Export/Backup all memories to a JSON file inside `data/memory/`.
- Add or delete memories for debugging purposes.

## Project Structure

```
ai-actor-project/
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
│   │   ├── stt/whisper_provider  ← OpenAI Whisper (EN/ES auto-detect)
│   │   ├── tts/kokoro_provider   ← Kokoro 82M ONNX (Ultra-fast local TTS)
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
ollama serve                   # Start in current terminal (Ctrl+C to stop)
pkill ollama                   # Stop from any terminal

# Check if running
pgrep -fl ollama               # Shows process if running
curl -s http://localhost:11434  # Returns "Ollama is running" if active

# Model management
ollama list                    # Show downloaded models
ollama pull <model>            # Download a model
ollama rm <model>              # Remove a model (frees disk space)
```

> **RAM note:** Ollama loads model weights into memory (~5-20GB depending on model). Models auto-unload after 5 minutes of idle. Use `ollama ps` to see what's currently loaded.

## Documentation

- [TODO.md](TODO.md) — Active development checklist
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full project requirements & open questions
- [TECH_STACK.md](TECH_STACK.md) — Technology research & model comparison
