# Tech Stack

> **Last updated**: 2026-08-19
> **Scope**: the technologies in play across the whole system — what is currently loaded, what is being compared against it, and what will slot in for the parts not yet built.
> **See also**: [ARCHITECTURE.md](ARCHITECTURE.md) for how these pieces are wired together and how data flows between them. This document is about *what* and *why*, not *how it connects*.

> ### ⚠️ No models are final
>
> **Model selection is still open across every layer of the stack.** What appears in `config/default.yaml` is what happens to be loaded for testing right now, not a decision. Evaluation is happening in the running system rather than on benchmarks — the question is how a model behaves in a real 72-hour aging performance in Rioplatense Spanish, which no leaderboard measures.
>
> Read every model below as *currently in testing*, and the notes as *observations so far*. The library and framework choices (Ollama, MLX, Chroma) are more settled than the models running on top of them, but even those remain swappable by design.

**Status legend**

| Mark | Meaning |
|------|---------|
| ✅ | Implemented and running — model within it still under evaluation |
| 🟡 | Dependency present, capability not fully applied |
| 🔜 | Not implemented — candidate technology identified |

---

## Evaluation constraints

Four constraints narrow the field at every layer. They are settled even though the models are not:

**Fully local.** No cloud calls at runtime. The piece runs 72 continuous hours in a museum; network dependence is an unacceptable failure mode. This rules out every hosted API regardless of quality.

**Apple Silicon.** The target machine is an M4 Pro with 48 GB unified memory. Preference goes to MLX-native and Metal-accelerated implementations.

**Rioplatense Spanish.** The performance language. Every model in the pipeline — STT, LLM, embeddings, TTS — is selected on Spanish quality, not English benchmarks. This eliminates several otherwise-strong English-centric options.

**Swappable by config.** Every capability sits behind an interface and is selected in `config/default.yaml`, never in code. This is what makes open-ended evaluation practical — trying a different model is a one-line change and a restart, not a refactor.

---

## The stack at a glance

Models listed here are **what is loaded for testing right now**, not settled choices.

| Layer | Technology | Currently loaded | RAM | Status |
|-------|-----------|-----------------|-----|--------|
| Audio capture | `sounddevice` | 16 kHz mono | — | ✅ |
| Voice activity | Energy gate (in-house) | — | — | ✅ |
| Speech-to-text | `mlx-whisper` | `whisper-small-mlx` | ~0.5 GB | ✅ |
| Vision | MediaPipe / deepface | — | ~1 GB | 🔜 |
| Inference engine | Ollama | HTTP, `localhost:11434` | — | ✅ |
| Language model | Gemma 4 | `gemma4:12b` | ~7 GB | ✅ |
| Embeddings | Qwen3 Embedding | `qwen3-embedding:0.6b` | ~1 GB | ✅ |
| Vector store | ChromaDB | `data/memory/` | ~0.5 GB | ✅ |
| Text-to-speech | Qwen3-TTS (MLX) | `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | ~2 GB | ✅ |
| Voice aging | `VoiceProfile` params | defined, unused | — | 🟡 |
| Audio playback | macOS `afplay` | subprocess | — | ✅ |
| Avatar / lip-sync | MuseTalk / ACTalker | — | ~4 GB | 🔜 |
| Runtime | Python | 3.13, venv | — | ✅ |
| Console | `rich` | — | — | ✅ |
| Tests | `pytest` + `pytest-asyncio` | asyncio auto-mode | — | ✅ |

**Today: ~11 GB. Full-system projection: ~16 GB.** On a 48 GB machine that leaves substantial headroom, which is why larger models stay live options — the binding constraint is per-turn latency, not memory.

---

## Perception layer

### Audio capture and voice activity ✅

`sounddevice` at 16 kHz mono, with an in-house energy-gate VAD (`energy_threshold 0.03`, `silence_duration_ms 800`, `preroll_ms 300`).

The VAD is deliberately naive. **Silero VAD** is the upgrade path if museum acoustics prove difficult — it is far more robust to ambient noise and crowd murmur, at the cost of a small model load and a few milliseconds per frame. Energy gating has been sufficient in testing, and the pre-roll window solves the one problem it reliably had (clipping soft first syllables).

### Speech-to-text ✅ — `mlx-whisper`

Model `small` → `mlx-community/whisper-small-mlx` (~244 MB), language pinned to `es`.

The provider maps friendly names to MLX community models: `tiny`, `base`, `small`, `medium`, `large`/`large-v3`, `turbo`. Moving up the ladder is a config edit.

**Why `mlx-whisper` is in place, and what it's being compared against:**

| Option | Notes so far |
|--------|--------------|
| **`mlx-whisper`** | **In use.** Native MLX, installs as a plain Python dependency, no build step or model conversion |
| `whisper.cpp` (V3 Turbo) | Likely faster via Core ML / ANE, but needs a separate build and GGML conversion. Untested here — worth trying if STT lands on the critical path |
| `faster-whisper` (CTranslate2) | Excellent on CUDA, no obvious Metal advantage |
| Distil-Whisper | Strong speed/accuracy trade, reportedly weaker on non-English |
| Vosk | Lightweight, expected to be materially worse on Spanish |

What favours it today is uniformity: the TTS stack is already MLX, so a second inference runtime would mean maintaining two toolchains. That is a practical argument rather than a measured one — no head-to-head against whisper.cpp has been run.

**On model size:** transcription happens every turn and sits directly in the latency path, so `small` is the current test point rather than a conclusion. `turbo` and `large-v3` are both worth a pass, particularly once the actress's real delivery is available to test against — rehearsal speech and performance speech differ more than model benchmarks do.

**Why language is pinned:** auto-detect misfires on short utterances and silently switches to Portuguese or Italian, which then poisons the LLM's language selection downstream. Forcing `es` costs nothing and removes a whole failure class.

### Vision 🔜

No camera or vision dependency is installed. Candidate stack when this is built:

| Task | Candidate | Notes |
|------|-----------|-------|
| Face detection | **MediaPipe** | Fast on Apple Silicon, mature, low RAM |
| Identity | InsightFace or `face_recognition` | Distinguish the actress from audience |
| Emotion | **deepface** | Multiple backends, easy to swap |
| Pose / gesture | MediaPipe | Body-language awareness |

Estimated ~1 GB combined. The architectural constraint on where vision output may be injected is significant and documented in [ARCHITECTURE.md](ARCHITECTURE.md) — it is not free to add.

---

## Cognition layer

### Inference engine ✅ — Ollama

HTTP API at `localhost:11434`, 10-minute client timeout for large-model cold starts.

| Option | Notes so far |
|--------|--------------|
| **Ollama** | **In use.** Model management, HTTP API, KV-cache reuse, trivial model swapping — the last point is what makes open evaluation cheap |
| MLX (direct) | Faster peak throughput; would lose the cache behaviour and model management the system currently leans on |
| llama.cpp | Maximum control, substantially more integration work |
| LM Studio | GUI-oriented, not designed to be driven programmatically |

Of everything in this document, this is the choice least likely to change — not because alternatives are worse, but because the whole evaluation workflow depends on swapping models in one line.

Two operational details that are not obvious:

`keep_alive: -1` is sent **per request**, not as an environment variable, because env vars do not reach the Ollama desktop app. It pins the model in memory so there is no cold start after a silence — essential when the actress may not speak for twenty minutes.

`OLLAMA_NUM_PARALLEL=2` must be set on the server (see `scripts/start_ollama.sh`, which installs via brew and sets it). Background memory-consolidation calls otherwise evict the conversation's KV-cache slot.

### Language model ✅ (model under evaluation) — currently `gemma4:12b`

~7 GB, dense, 128K context. Parameters: `temperature 0.8`, `max_tokens 1024`, `streaming true`, `think false`.

**This is the most open question in the stack.** The LLM carries the character, so it is the layer where in-system testing matters most and where benchmark scores are least useful. Several models are in rotation.

**Models tried so far on this machine** — mirrored in `config/default.yaml`:

| Model | Size | Type | Context | Respects `think:false` | Observations |
|-------|------|------|---------|------------------------|--------------|
| **`gemma4:12b`** | ~7 GB | dense | 128K | yes | Currently loaded — good latency, holds character well so far |
| `gemma4:26b` | ~15 GB | MoE | 256K | yes | Next step up if quality needs it; RAM is available |
| `qwen3.5:9b` | 6.6 GB | dense | 262K | yes | Strong multilingual, character voice reads flatter |
| `qwen3.5:27b` | 17 GB | dense | 262K | yes | Good output, noticeably slower per turn |
| `qwen3:30b-a3b` | 18 GB | MoE | 128K | **no** | Set aside — thinking leaks into output despite the flag |
| `llama3.1:8b` | 5 GB | dense | 128K | n/a | Fast option for quick prototyping runs |

**What we're evaluating on**, roughly in priority order: Spanish fluency and register (Rioplatense specifically, not neutral Latin American), ability to hold a character across long sessions, per-turn latency, and honouring `think: false`.

That last one has the clearest practical consequence. Models ignoring the flag emit reasoning blocks that must be stripped at stream level, which costs latency and occasionally leaks fragments into synthesized speech — which is why `qwen3:30b-a3b` is not currently in rotation despite otherwise reasonable output.

**Not yet tried:** very large MoE models (Llama 4 Maverick, Qwen3 235B-A22B, DeepSeek V3.2) were considered on paper in early research and never tested, since the 12B–27B range has been adequate so far and latency matters more to the piece than marginal quality gains. Roleplay-tuned models such as MN Violet Lotus were skipped on expected Spanish weakness rather than on testing. Both groups remain open if the current range proves insufficient.

### Embeddings ✅ (model under evaluation) — currently `qwen3-embedding:0.6b`

Served by the same Ollama instance, ~1 GB.

| Option | Notes so far |
|--------|--------------|
| **`qwen3-embedding:0.6b`** | **Currently loaded.** Best sub-1 GB multilingual found so far, real Spanish capability, no extra runtime |
| Chroma default (`all-MiniLM-L6-v2`) | Tried and moved away from — English-centric, retrieves poorly against Spanish memories |
| `nomic-embed-text` | Convenient, still English-first |
| `BAAI/bge-large-en-v1.5` | Higher quality, English only, and adds a second runtime |

Switching costs more here than elsewhere in the stack, which is a reason to be deliberate rather than a reason it is settled — see the migration hazard below.

> **Migration hazard:** embeddings from different models are mathematically incompatible. Changing `memory.embedding_model` requires deleting `data/memory/` or setting a new `collection_name`. There is no in-place re-index.

### Vector store ✅ — ChromaDB

Persisted to `data/memory/`, collection `ai_actor_memories`, retrieval limit 5 per turn.

| Option | Notes so far |
|--------|--------------|
| **ChromaDB** | **In use.** Simplest embedded persistence, pluggable embedding functions, adequate at this scale |
| LanceDB | Faster and more modern; worth revisiting only if the store grows well beyond a single performance |
| Qdrant | Production-grade, but wants a server process — contrary to the single-process local design |

Scale is genuinely small: a 72-hour run produces on the order of thousands of consolidated memories, not millions. Chroma's performance ceiling is nowhere near being a constraint. A keyword-matching `simple` provider exists as a zero-dependency fallback.

---

## Expression layer

### Text-to-speech ✅ (model under evaluation) — currently Qwen3-TTS via MLX

`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, ~1.8 GB, voice `serena`, language `es`.

Voice quality is an active workstream — this layer is under as much scrutiny as the LLM, and for the same reason: it is judged by ear in the room, not by a metric.

| Option | Latency | Emotion control | Notes so far |
|--------|---------|-----------------|--------------|
| **Qwen3-TTS 1.7B** | ~97 ms first audio | Natural-language instruction | **Currently loaded** |
| Qwen3-TTS 0.6B | Faster | Same mechanism | Untested here (~700 MB) — the move if latency becomes critical |
| Chatterbox-Turbo | <150 ms | `[laugh]`-style tags | Strong candidate, MIT, 350M params |
| IndexTTS 2.0 | ~400 ms | Disentangled emotion/identity | Worth testing if emotion and timbre need independent control |
| CosyVoice2 | ~150 ms | Dialect control | Viable, less mature tooling |
| F5-TTS | ~200 ms | Moderate | Good cloning, weaker expressive range |
| Bark | Slow | Excellent non-verbals | Too slow for real-time |
| XTTS v2 | Moderate | Moderate | Superseded; licence complications |

**What makes Qwen3-TTS worth testing first:** delivery is steered by a natural-language instruction string rather than a fixed emotion enum. For a character whose emotional range must widen across sixty years of aging, an open-ended instruction channel promises more than a faster model with five preset moods. Cross-lingual voice cloning from a ~3-second reference would also make the voice-aging plan tractable. Both remain to be proven in performance conditions.

**Known hazard:** emojis, inline `[tags]`, and `*actions*` in the input derail this model into wrong-language output — Spanish text containing an emoji has produced CJK speech, verified by closed-loop probing. All TTS input is sanitized before synthesis; see [ARCHITECTURE.md](ARCHITECTURE.md) for the contract.

Built-in speakers — female: `serena`, `vivian`, `aura`, `sohee`, `ono_anna`; male: `ryan`, `aiden`, `eric`, `dylan`, `uncle_fu`.

**Fallback provider:** `system` (macOS `say`), config-selectable and genuinely useful mid-performance. If the Qwen model fails to load during a run, dropping to it keeps the piece going with a degraded voice instead of silence — and because it has no model to load and no dependency to install, it cannot fail the same way.

### Voice aging 🟡

`VoiceProfile` already carries `pitch_shift` (semitones), `speed`, `breathiness`, and `tremor`, and `profiles/age_XX_YY/` is the intended home for per-stage `voice_reference.wav` files.

**None of it is applied.** All eight age stages currently speak as `serena` at `speed 1.0`. She ages in language, memory, and personality — not in timbre.

| Stage | Pitch | Rate | Timbre |
|-------|-------|------|--------|
| 10-15 | Higher | Faster, variable | Lighter |
| 15-20 | Transitioning | Varied | Developing |
| 20-25 | Adult range | Moderate | Clear |
| 25-30 | Stable | Confident | Full |
| 30-40 | Stable | Measured | Rich |
| 40-50 | Slightly lower | Slower | Deeper |
| 50-60 | Lower | Slower | Rougher |
| 60-70 | Lower | Deliberate | Aged |

Two candidate routes: apply the existing `VoiceProfile` parameters as DSP post-processing on Qwen3-TTS output (cheap, less convincing), or record eight reference clips and use Qwen3-TTS cloning per stage (better, needs a voice actor or a synthesis pass). Transitions should crossfade over roughly 30 minutes — an abrupt switch at a stage boundary reads as a bug rather than as aging.

### Audio playback ✅ — macOS `afplay`

`WavPlayback` writes synthesized audio to a temp WAV and plays it as an async subprocess, output rate 22050 Hz.

This is the most platform-bound component in the stack. Porting to Linux means replacing this one class — `sounddevice` playback or `ffplay` would both serve.

### Avatar / lip-sync 🔜

Not implemented, and planned as a **separate system fed a data feed** rather than an in-process provider, so rendering never competes with the LLM for memory or scheduling.

| Candidate | Realism | Real-time | Notes |
|-----------|---------|-----------|-------|
| **MuseTalk** | Good | Yes, 30+ fps | Proven real-time lip-sync, simplest integration |
| **ACTalker** | Excellent | Yes | Audio + motion driven; could relay the actress's own expressions |
| Sonic | Excellent | Near real-time | Natural idle motion and breathing |
| LivePortrait | Highest | No | Needs a driving video |
| SadTalker | Good | No | Pre-rendering only |

Estimated ~4 GB. Face references would live at `profiles/age_XX_YY/face_reference.png`, matching the voice-reference convention. An `AvatarProvider` interface was drafted in earlier versions of this document but was **never implemented and is not in the codebase** — if the avatar remains a separate process, it should stay that way.

---

## Runtime and tooling

| Component | Choice | Notes |
|-----------|--------|-------|
| Python | 3.13, `venv` | `pip install -e ".[whisper,tts,memory,dev]"` |
| Config | PyYAML | Layered: `default.yaml` → `<env>.yaml` → `AI_ACTOR__*` env vars |
| HTTP | `httpx` | Async client for Ollama |
| Audio I/O | `sounddevice`, `soundfile`, `numpy` | Capture and WAV handling |
| Console | `rich` | Streaming display, tables, status lines |
| Tests | `pytest`, `pytest-asyncio` | asyncio auto-mode — write `async def test_*`, no marker |
| Benchmarking | `psutil` + `scripts/benchmark.py` | Per-model latency and memory |
| Memory inspection | Gradio (`scripts/chroma_explorer.py`) | Web UI at `:7860`, or `--cli` for a Rich table |

No linter or formatter is configured.

**Platform lock:** macOS on Apple Silicon. `afplay` for playback, MLX for both STT and TTS. Ollama must be running.

---

## Dependency licensing

Relevant when deciding how this repository itself may be licensed and redistributed.

| Component | Licence | Note |
|-----------|---------|------|
| Ollama | MIT | Engine only |
| **Gemma 4 weights** | **Gemma Terms of Use** | **Not OSI open source — carries use restrictions.** Verify before any redistribution |
| Whisper | MIT | OpenAI, permissive |
| MLX / `mlx-whisper` | MIT | Apple |
| Qwen3-TTS | Apache 2.0 | Verify for the specific MLX community conversion |
| Qwen3 Embedding | Apache 2.0 | |
| ChromaDB | Apache 2.0 | |

> ⚠️ Verify each of these against the current upstream licence text before publishing or distributing. Model licences change between releases, and the Gemma terms in particular are more restrictive than the permissive licences elsewhere in the stack. Note that the repository ships no weights — models are pulled by the user at setup — which limits exposure considerably.

---

## Swapping and extending

Provider selection is configuration, never code:

```yaml
llm:
  provider: "ollama"
  model: "gemma4:12b"
```

```bash
AI_ACTOR__LLM__MODEL=qwen3.5:27b python -m src.main
AI_ACTOR_ENV=development python -m src.main
python -m src.main --list-providers
```

Adding a new technology takes two steps: implement the interface under `src/providers/<kind>/`, then add one line to the matching dict in `src/core/registry.py`. Nothing else changes. Currently registered:

```python
_STT_PROVIDERS    = {"whisper"}
_LLM_PROVIDERS    = {"ollama"}
_TTS_PROVIDERS    = {"system", "qwen"}
_MEMORY_PROVIDERS = {"chroma", "simple"}
```

---

## Planned additions

| Capability | Candidate technology | RAM | Blocking on |
|-----------|---------------------|-----|-------------|
| Vision 🔜 | MediaPipe + deepface | ~1 GB | Camera hardware decision |
| Avatar 🔜 | MuseTalk or ACTalker | ~4 GB | Projection/smoke-screen test |
| Voice aging 🟡 | Qwen3-TTS cloning, 8 references | — | Reference recordings |
| Robust VAD 🔜 | Silero VAD | ~50 MB | Only if venue acoustics demand it |
| Importance scoring 🟡 | — | — | Retrieval currently ranks on similarity alone |
| Watchdog 🔜 | `launchd` or supervisor | — | Needed before any unattended 72-hour run |

---

*Every component here is swappable by configuration. The AI landscape moves monthly; the interfaces are the part meant to outlive the model choices.*
