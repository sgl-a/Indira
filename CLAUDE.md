# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal, **fully-local** AI theatrical performance system. An artificial being ("Indira") ages from 10 to 70 over a 72-hour durational installation, conversing in real time with a live actress ("Ximena", her "mother") and plausible others. Everything runs on-device (Ollama + MLX on Apple Silicon) — no cloud calls at runtime. 

The performance language is **Spanish (Rioplatense)**. Character identity, personality prompts, and stored memory strings are written in Spanish on purpose — this is a domain requirement, not an oversight. **Don't "fix" Spanish strings to English.**

## Implementation status — read the code, not the prose

This project is early-stage WIP. The Markdown docs describe the *originally designed* full system; much of it is **not built yet**. The docs have been reconciled with reality using status tags — **trust those tags and the code over aspirational prose**:
- ✅ **Working today:** audio/text conversational actor — mic→STT→LLM→TTS→speaker, memory, 8-stage aging personality, emotion tags.
- 🔜 **Designed but NOT built:** computer vision / cameras, the visual avatar (lip-sync/projection — planned as a *separate* system fed a data feed), proactive conversation, 72h reliability/watchdog.
- 🟡 **Partial:** age-based voice change (designed, not applied — one voice today), memory importance scoring (fixed `0.5`).

See [ARCHITECTURE.md](ARCHITECTURE.md) (Implementation Status table) and [TODO.md](TODO.md) for the authoritative breakdown.

## Commands

```bash
# Setup (Python 3.13 + venv)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[whisper,tts,memory,dev]"   # extras: whisper, tts, memory, dev

# Ollama must be running for the LLM (separate terminal)
ollama serve
ollama pull gemma4:12b        # the currently configured default LLM

# Run
python -m src.main                 # text mode (default, no mic needed)
python -m src.main --mode audio    # voice mode (mic → STT → LLM → TTS)
python -m src.main --fresh         # ignore saved performance state, start at hour 0
# (performance timing persists to data/performance_state.json and the conversation
#  transcript to data/conversation_transcript.jsonl — both auto-resume after a
#  crash/restart if still inside the 72h window)
python -m src.main --list-providers
python -m src.main --config config --env development   # loads config/development.yaml overrides

# Benchmarks & memory tooling
python -m scripts.benchmark llm --models llama3.1:8b,qwen3.5:27b
python -m scripts.benchmark full
python scripts/chroma_explorer.py         # Gradio web UI at :7860
python scripts/chroma_explorer.py --cli   # Rich table in terminal
```

**Tests:** run with `pytest` (venv). `tests/` covers the text-processing and persistence seams: emotion-tag parsing, incremental `<think>` stripping, memory recency, performance-state resume. Write `async def test_*` — asyncio auto-mode means no `@pytest.mark.asyncio` needed. No linter/formatter is configured.

**Runtime notes:** audio playback uses macOS `afplay` and STT uses `mlx-whisper` — this is a **macOS / Apple-Silicon-only** runtime. `main()` ends with `os._exit(0)` to force-kill the blocking `input()` executor thread; a normal return won't shut the process down cleanly (this also causes a benign leaked-semaphore warning at exit — see TODO.md).

## Architecture

Built on a **provider pattern**: every swappable capability (STT, LLM, TTS, Memory) is an abstract interface with concrete implementations selected *by config, not code*.

```
Mic/text → STT → Orchestrator ⇄ LLM → TTS → speaker (afplay)
                      ⇅
              Memory + AgeEngine + ActorState
```

- `src/core/interfaces/` — the four ABCs (`stt.py`, `llm.py`, `tts.py`, `memory.py`) + dataclasses (`LLMResponse`, `Memory`, `VoiceProfile`, `TranscriptionResult`). All provider methods are `async`.
- `src/core/registry.py` — maps provider names → dotted class paths and lazily imports them. **This is the extension point.** To add a provider: implement the interface under `src/providers/<kind>/`, then add one line to the relevant `_*_PROVIDERS` dict here (or call `register_provider(...)`). No other code changes.
- `src/core/orchestrator.py` — the heart. Conversation loop (`process_input`, `process_input_streaming`), memory retrieval/storage and background consolidation, between-turns maintenance (window trim + KV-cache prewarm), the interactive `run_text_mode` / `run_voice_mode` loops, slash-command handling, and the overlapping streaming-TTS pipeline.
- `src/core/age_engine.py` — maps `hours_elapsed` → one of 8 age stages, loads that stage's `profiles/age_XX_YY/personality.yaml`, and **builds the LLM system prompt** (`build_personality_prompt`) by combining permanent identity (from config) with age-specific traits (from profile YAML).
- `src/core/state.py` — `ActorState`, the single source of truth (age stage, emotion, conversation history, timing). `hours_elapsed` derives from `performance_start_time`; the `/age N` command "time-travels" by rewriting that start time.
- `src/core/config.py` — loads `config/default.yaml`, deep-merges `config/<env>.yaml`, then applies env-var overrides. Module-level singleton via `get_config()`.

### Key cross-cutting conventions

- **Config is the control surface.** `config/default.yaml` drives everything. Override precedence: `default.yaml` → `<env>.yaml` (via `INDIRA_ENV` or `--env`) → env vars of the form `INDIRA__SECTION__KEY=value` (double underscores, e.g. `INDIRA__LLM__MODEL=qwen3.5:27b`).
- **Emotion-tag protocol.** The LLM is prompted (in `age_engine.build_personality_prompt`) to prefix each reply with `[emoción] texto`. `ollama_provider._parse_emotion` and the orchestrator's streaming path parse/strip that leading bracket tag to extract `current_emotion`, which is then mapped to a TTS delivery instruction. Preserve this contract on both the prompt side and any parsing side.
- **Prompt-cache contract.** Ollama reuses its KV cache for a byte-identical prompt prefix, so the system prompt (`age_engine.build_personality_prompt`) must stay **byte-stable within an age stage** — no per-turn content in it. Everything volatile (retrieved memories) rides in the `[Contexto ...]` envelope prepended to the *newest* user message (`orchestrator._wrap_with_context`); history stores clean text so replay stays prefix-stable. Emotion is **output-only** (LLM → `state.current_emotion` → TTS/console) — never inject it back into the prompt; continuity comes from `get_recent_messages` re-prefixing `[emoción]` tags onto replayed assistant turns.
- **TTS input must be sanitized.** Emojis, inline `[tags]`, and `*actions*` derail Qwen3-TTS into wrong-language babble (verified by closed-loop probing). All text reaching a TTS provider goes through `text_filters.sanitize_for_tts()` first; skip synthesis when it returns `""`. The console still shows the original text.
- **Memory pipeline.** Short-term = the replayed transcript window (grows to `memory.history_max_turns`, block-trims to `history_trim_to`; persisted to `data/conversation_transcript.jsonl`, auto-restored on resume). Long-term = `Memory` objects (`age_stage`, `emotional_tag`, `importance`, `memory_type` `consolidated` | `milestone`) retrieved by semantic `search()` only. **Consolidation** (`orchestrator._consolidate_block`): turns dropped from the window are compressed in the background — while idle — by the LLM *as the character at the age the block happened* into 0–3 first-person memories with self-scored importance; boring blocks store nothing; unparseable output falls back to a mechanical summary. Age transitions auto-store a `milestone`. `chroma` provider = semantic (vector DB in `data/memory/`, embedder set by `memory.embedding_model` — switching embedders needs a fresh collection); `simple` = keyword fallback. Run Ollama with `OLLAMA_NUM_PARALLEL=2` so consolidation calls don't evict the conversation's KV-cache slot.
- **Ollama specifics.** `think: false` in config disables model thinking (Qwen 3.5 respects it, Qwen3 ignores it); `ollama_provider._strip_thinking` removes any leaked `<think>` blocks. HTTP timeout is 10 min for large models. Model pinning is per-request (`llm.keep_alive: -1` in config — env vars don't reach the desktop app). Run the server via `scripts/start_ollama.sh` (brew install; sets `OLLAMA_NUM_PARALLEL=2` so consolidation calls don't evict the conversation's KV-cache slot).

## Large / ignored assets

`.gitignore` excludes `models/`, `data/`, `.venv/`, all audio/video, and `*.egg-info/`. Age profiles under `profiles/` are the exception — the personality YAML is tracked; referenced `voice_reference.wav` / `face_reference.png` media are not.
