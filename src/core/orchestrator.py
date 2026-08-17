from __future__ import annotations

"""
Orchestrator.

The central coordinator that connects all providers and manages
the AI Actor's conversation loop. Handles:
- Audio input → STT → LLM → TTS → Audio output
- Memory storage, retrieval, and background consolidation
- Age progression
"""

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from src.core.audio_output import WavPlayback
from src.core.logging_utils import defer_console_logs, flush_console_logs
from src.core.text_filters import EmotionTagFilter, sanitize_for_tts
from src.core.state import ActorState, ConversationTurn, PerformancePhase
from src.core.age_engine import AgeEngine
from src.core.registry import (
    create_llm_provider,
    create_stt_provider,
    create_tts_provider,
    create_memory_provider,
)
from src.core.interfaces.llm import LLMProvider, LLMResponse
from src.core.interfaces.stt import STTProvider
from src.core.interfaces.tts import TTSProvider, VoiceProfile
from src.core.interfaces.memory import MemoryProvider, Memory

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main coordinator for the AI Actor system.

    Manages the perception → cognition → output pipeline
    and handles all inter-component communication.
    """

    def __init__(self, config: dict):
        self.config = config
        self.state = ActorState()
        self.age_engine = AgeEngine(config)

        # Providers (initialized in setup)
        self.stt: STTProvider | None = None
        self.llm: LLMProvider | None = None
        self.tts: TTSProvider | None = None
        self.memory: MemoryProvider | None = None

        # Control
        self._running = False

        # Performance-state persistence (crash recovery for the 72h run)
        system_config = config.get("system", {})
        self._state_file = Path(system_config.get("state_file", "data/performance_state.json"))
        self._performance_duration_hours = system_config.get("performance_duration_hours", 72)
        # Transcript persistence: conversation_history lives in RAM, so
        # without this a mid-show crash would resume the right age and
        # long-term memories but an empty short-term window
        self._transcript_file = Path(
            system_config.get("transcript_file", "data/conversation_transcript.jsonl")
        )

        # Short-term window + memory consolidation (see _consolidate_block)
        memory_config = config.get("memory", {})
        self._history_max_turns = memory_config.get("history_max_turns", 80)
        self._history_trim_to = memory_config.get("history_trim_to", 60)
        consolidation_config = memory_config.get("consolidation", {})
        self._consolidation_enabled = consolidation_config.get("enabled", True)
        self._consolidation_temperature = consolidation_config.get("temperature", 0.3)
        self._consolidation_max_tokens = consolidation_config.get("max_tokens", 400)
        self._consolidation_queue: asyncio.Queue[list[ConversationTurn]] = asyncio.Queue()
        self._consolidation_task: asyncio.Task | None = None

    def _start_or_resume_performance(self) -> bool:
        """
        Start the performance, resuming a previous run if a saved state
        exists and is still inside the performance window.

        Without this, a crash at hour 50 would restart the character at
        age 10 with an old woman's memories. Returns True if resumed.
        """
        fresh = self.config.get("system", {}).get("fresh_start", False)

        if not fresh and self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                start = float(data["performance_start_time"])
                elapsed_hours = (time.time() - start) / 3600
                if 0 <= elapsed_hours < self._performance_duration_hours:
                    self.state.performance_start_time = start
                    self.state.performance_phase = PerformancePhase.RUNNING
                    self.state.last_interaction_time = time.time()
                    self.age_engine.update_state(self.state)
                    restored_turns = self._load_transcript()
                    # Restore the window position so already-consolidated
                    # turns aren't re-consolidated (duplicate memories) or
                    # replayed into the prompt
                    self.state.history_window_start = min(
                        int(data.get("history_window_start", 0)), restored_turns
                    )
                    logger.info(
                        f"⏮️  Resumed performance at hour {elapsed_hours:.1f} "
                        f"(age stage {self.state.current_age_stage}, "
                        f"{restored_turns} transcript turns) from {self._state_file}"
                    )
                    return True
                logger.info(
                    f"Saved performance state is {elapsed_hours:.1f}h old "
                    f"(window is {self._performance_duration_hours}h) — starting fresh"
                )
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Could not read {self._state_file}: {e} — starting fresh")

        self.state.start_performance()
        self._persist_performance_state()
        # New performance, new transcript
        try:
            self._transcript_file.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not clear old transcript: {e}")
        return False

    def _append_transcript(self, turn: ConversationTurn) -> None:
        """Append one turn to the on-disk transcript (JSONL, crash-safe:
        a torn last line is skipped on load)."""
        try:
            self._transcript_file.parent.mkdir(parents=True, exist_ok=True)
            with self._transcript_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "role": turn.role,
                    "content": turn.content,
                    "emotion": turn.emotion,
                    "timestamp": turn.timestamp,
                    "speaker_id": turn.speaker_id,
                }, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Could not append to transcript: {e}")

    def _load_transcript(self) -> int:
        """Restore conversation history from the on-disk transcript.
        Returns the number of turns restored."""
        if not self._transcript_file.exists():
            return 0
        turns: list[ConversationTurn] = []
        try:
            lines = self._transcript_file.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning(f"Could not read transcript {self._transcript_file}: {e}")
            return 0
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                turns.append(ConversationTurn(
                    role=data["role"],
                    content=data["content"],
                    emotion=data.get("emotion"),
                    timestamp=data.get("timestamp", 0.0),
                    speaker_id=data.get("speaker_id"),
                ))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # torn line from a crash mid-write
        self.state.conversation_history = turns
        return len(turns)

    def _persist_performance_state(self) -> None:
        """Write performance timing to disk (atomically) for crash recovery."""
        if self.state.performance_start_time is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "performance_start_time": self.state.performance_start_time,
                "history_window_start": self.state.history_window_start,
                "saved_at": time.time(),
            }))
            tmp.replace(self._state_file)
        except OSError as e:
            logger.warning(f"Could not persist performance state: {e}")

    async def setup(self) -> None:
        """Initialize all providers concurrently.

        Startup time = the slowest provider (the TTS model load), not the
        sum: each provider's heavy work runs in its own executor thread, so
        the four initializations genuinely overlap.
        """
        logger.info("🎭 Initializing AI Actor system...")
        start = time.time()

        llm, tts, memory, stt = await asyncio.gather(
            create_llm_provider(self.config),
            create_tts_provider(self.config),
            create_memory_provider(self.config),
            create_stt_provider(self.config),
            return_exceptions=True,
        )

        # STT is optional — text mode runs without a mic stack
        if isinstance(stt, Exception):
            logger.warning(f"STT initialization failed: {stt}. Running in text-only mode.")
            stt = None
        self.stt = stt

        # The rest are required. Assign whatever succeeded FIRST so
        # shutdown() can clean up a partial initialization, then fail loudly.
        self.llm = llm if not isinstance(llm, Exception) else None
        self.tts = tts if not isinstance(tts, Exception) else None
        self.memory = memory if not isinstance(memory, Exception) else None
        for name, result in (("LLM", llm), ("TTS", tts), ("Memory", memory)):
            if isinstance(result, Exception):
                raise RuntimeError(f"{name} provider failed to initialize: {result}") from result

        # Background memory consolidation (turns dropped from the short-term
        # window get compressed into long-term memories while she's idle)
        if self._consolidation_enabled and self.llm and self.memory:
            self._consolidation_task = asyncio.create_task(self._consolidation_worker())

        logger.info(f"✅ All providers initialized in {time.time() - start:.1f}s")

    async def shutdown(self) -> None:
        """Clean shutdown of all providers."""
        logger.info("Shutting down AI Actor system...")
        self._running = False

        if self._consolidation_task:
            self._consolidation_task.cancel()

        for provider in [self.stt, self.llm, self.tts, self.memory]:
            if provider:
                try:
                    await provider.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {provider.__class__.__name__}: {e}")

        logger.info("👋 AI Actor system shut down")

    async def process_input(self, user_text: str, speaker_id: str | None = None) -> str:
        """
        Process text input and return the complete response text.

        Thin non-streaming wrapper: the single turn implementation lives in
        process_input_streaming — this just consumes it and keeps the final
        text (all state updates, memory, and maintenance happen there).
        """
        response = None
        async for chunk in self.process_input_streaming(user_text, speaker_id=speaker_id):
            if isinstance(chunk, LLMResponse):
                response = chunk
        return response.text if response else ""

    async def process_input_streaming(
        self, user_text: str, speaker_id: str | None = None
    ) -> AsyncIterator[str | LLMResponse]:
        """
        THE conversation turn: record input, retrieve memories, build the
        prompt, stream the response, update state/memory, run maintenance.

        Yields str chunks as they arrive, then a final LLMResponse.
        The caller should display chunks in real-time, then use the
        LLMResponse for metadata (emotion, timing).
        (Non-streaming callers use the process_input wrapper.)
        """
        if not self.llm:
            raise RuntimeError("LLM provider not initialized")

        # Update age progression
        age_changed = self.age_engine.update_state(self.state)
        if age_changed:
            await self._store_age_milestone()

        # 1. Record user input
        self._append_transcript(self.state.add_turn("user", user_text, speaker_id=speaker_id))

        # 2. Retrieve relevant memories
        memories_context = await self._build_memory_context(user_text)

        # 3. Build system prompt (stable per age stage — Ollama KV-cache friendly)
        system_prompt = self.age_engine.build_personality_prompt(self.state, self.config)

        # 4. Get conversation history. (Safety-net trim only: the window is
        # normally trimmed between turns by _post_turn_maintenance so the
        # KV re-prefill happens while idle, not on a live turn.) Volatile
        # context (memories) rides on the newest message only — the stored
        # turn keeps the clean text so replay stays byte-stable for the cache
        self._queue_dropped_turns(
            self.state.trim_history(self._history_max_turns, self._history_trim_to)
        )
        messages = self.state.get_recent_messages()
        messages[-1]["content"] = self._wrap_with_context(user_text, memories_context)

        # 5. Stream response (part of the LLMProvider interface — providers
        # without real token streaming inherit a generate() fallback)
        llm_config = self.config.get("llm", {})
        response = None

        async for chunk in self.llm.stream_generate_with_metadata(
            system_prompt=system_prompt,
            messages=messages,
            temperature=llm_config.get("temperature", 0.8),
            max_tokens=llm_config.get("max_tokens", 512),
        ):
            if isinstance(chunk, LLMResponse):
                response = chunk
            else:
                yield chunk

        # 6. Post-streaming: update state and store memory
        if response:
            self._append_transcript(
                self.state.add_turn("assistant", response.text, emotion=response.emotion)
            )
            if response.emotion:
                self.state.current_emotion = response.emotion
            # With consolidation on, long-term memories are written when
            # turns leave the short-term window, not per turn
            if not self._consolidation_enabled:
                await self._store_interaction(user_text, response)
            # Between-turns maintenance (trim + cache prewarm, runs in the gap)
            self._post_turn_maintenance()
            logger.debug(
                f"💬 Response ({response.generation_time_ms:.0f}ms, "
                f"{response.tokens_generated} tokens, "
                f"emotion={response.emotion}): {response.text[:80]}..."
            )

        # Yield final response for caller to use if needed
        if response:
            yield response

    def _get_voice_profile(self) -> VoiceProfile:
        """Get the voice profile for the current age stage."""
        stage = self.age_engine.get_current_stage(self.state)
        return VoiceProfile(
            id=f"voice_{stage.range}",
            name=f"Voice {stage.range}",
            age_stage=stage.range,
            reference_audio_path=stage.voice_profile_path or "",
        )

    def _wrap_with_context(self, user_text: str, memories_context: str) -> str:
        """
        Prepend the per-turn volatile context (memories) to the outgoing
        user message.

        This content changes every turn, so it lives at the tail of the
        request — the final message is new tokens regardless — instead of
        the system prompt, where any changed byte would invalidate Ollama's
        KV cache for the entire prompt + history behind it.

        Emotion is deliberately NOT injected here: it is an output of each
        turn (LLM → state → TTS/console/face), and emotional continuity
        comes from the [emoción] tags replayed in history — see
        ActorState.get_recent_messages.
        """
        if not memories_context:
            return user_text
        return f"[Contexto — no es parte del diálogo:\n{memories_context}]\n\n{user_text}"

    async def _build_memory_context(self, query: str) -> str:
        """
        Build memory context string for the LLM prompt.

        Long-term recall only: semantic search across the whole performance.
        Short-term context is the replayed transcript (get_recent_messages),
        which persists via _append_transcript — recent memories from the
        store would just duplicate it.
        """
        if not self.memory:
            return ""

        memories = await self.memory.search(
            query, limit=self.config.get("memory", {}).get("long_term_search_limit", 5)
        )
        if not memories:
            return ""

        parts = ["Cosas que recordás:"]
        for mem in memories:
            emotion_str = f" (sintiendo {mem.emotional_tag})" if mem.emotional_tag else ""
            parts.append(f"- [Edad {mem.age_stage}]{emotion_str}: {mem.content}")

        logger.debug(f"📝 Memory context ({len(memories)} memories injected into prompt)")
        return "\n".join(parts)

    async def _store_interaction(self, user_input: str, response: LLMResponse) -> None:
        """Store a significant interaction in memory."""
        if not self.memory:
            return

        # Include both what was said and what was responded —
        # this is what makes semantic search actually work.
        name = self.config.get("llm", {}).get("personality", {}).get("name", "Indira")
        user_part = user_input[:150].strip()
        response_part = response.text[:200].strip()
        summary = f"Mamá dijo: {user_part}. {name} respondió: {response_part}"

        memory = Memory(
            id=str(uuid.uuid4()),
            content=summary,
            age_stage=self.state.current_age_stage,
            emotional_tag=response.emotion,
            importance=0.5,
            memory_type="interaction",
        )
        await self.memory.store(memory)

    # ─── Memory consolidation ───────────────────────────────────────────
    # When turns leave the short-term window they get compressed into a few
    # long-term memories, written by the LLM in the character's own voice at
    # her current age, each with a self-scored importance. Runs in the
    # background while she's idle so it never delays a live response.

    def _post_turn_maintenance(self) -> None:
        """
        Runs after each completed turn, so heavy bookkeeping lands in the
        idle gap between turns instead of delaying a live response.

        Trims the window one exchange EARLY (max_turns - 1): the next user
        turn would push it over the limit, and trimming on that turn would
        change the request prefix mid-conversation — a full KV re-prefill
        (seconds of extra first-token latency) paid on a live turn. Trimming
        now and pre-warming the cache in the background makes the re-prefill
        invisible instead.
        """
        dropped = self.state.trim_history(self._history_max_turns - 1, self._history_trim_to)
        if dropped:
            self._queue_dropped_turns(dropped)
            asyncio.create_task(self._prewarm_cache())

    async def _prewarm_cache(self) -> None:
        """Re-prefill Ollama's KV cache with the post-trim prefix (system
        prompt + trimmed window) so the next live turn starts from a warm
        cache. Waits until she's idle; the 1-token generation is discarded."""
        if not self.llm:
            return
        while self.state.is_speaking:
            await asyncio.sleep(0.5)
        try:
            system_prompt = self.age_engine.build_personality_prompt(self.state, self.config)
            messages = self.state.get_recent_messages()
            if not messages:
                return
            await self.llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.0,
                max_tokens=1,
            )
            logger.debug("🔥 KV cache pre-warmed with post-trim prefix")
        except Exception as e:
            logger.debug(f"Cache prewarm failed (harmless, next turn just re-prefills): {e}")

    def _queue_dropped_turns(self, dropped: list[ConversationTurn]) -> None:
        """Hand turns that fell out of the short-term window to the
        consolidation worker, and persist the new window position."""
        if not dropped:
            return
        self._persist_performance_state()
        if self._consolidation_enabled and self.llm and self.memory:
            self._consolidation_queue.put_nowait(dropped)
            logger.debug(f"🧠 Queued {len(dropped)} turns for consolidation")

    async def _consolidation_worker(self) -> None:
        """Background task: consolidate dropped blocks whenever she's idle."""
        while True:
            block = await self._consolidation_queue.get()
            # Wait for the current turn (generation + speech) to finish so
            # the consolidation call never competes with a live response
            while self.state.is_speaking:
                await asyncio.sleep(0.5)
            try:
                await self._consolidate_block(block)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Consolidation failed ({e}); storing mechanical fallback")
                await self._store_block_fallback(block)

    async def _consolidate_block(self, block: list[ConversationTurn]) -> None:
        """Compress a block of dropped turns into 0-3 long-term memories."""
        if not (self.llm and self.memory and block):
            return

        # Age stage at the time the block happened (not necessarily now)
        stage = self._stage_for_timestamp(block[-1].timestamp)
        name = self.config.get("llm", {}).get("personality", {}).get("name", "Indira")

        transcript = "\n".join(
            f"{'Mamá' if turn.role == 'user' else name}: {turn.content}"
            for turn in block
        )

        system_prompt = (
            f"Sos {name}, tenés {stage.range} años. Este fragmento de una charla con tu mamá "
            f"está por desvanecerse de tu memoria inmediata. Escribí los recuerdos que te "
            f"quedan de él: entre cero y tres.\n"
            f"\n"
            f"Reglas:\n"
            f"- Cada recuerdo: una o dos frases, en primera persona, con tus palabras de "
            f"ahora, a tus {stage.range} años.\n"
            f"- Guardá solo lo que importa. Importancia 0.7 a 1.0: revelaciones, emociones "
            f"fuertes, promesas, peleas, cosas nuevas sobre tu mamá o sobre vos. "
            f"0.4 a 0.6: una charla significativa. Menos que eso no se recuerda.\n"
            f"- Si no pasó nada que valga la pena recordar, respondé únicamente: NADA\n"
            f"- Formato: una línea JSON por recuerdo, sin ningún otro texto:\n"
            f'{{"recuerdo": "...", "emocion": "...", "importancia": 0.0}}'
        )

        response = await self.llm.generate(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": transcript}],
            temperature=self._consolidation_temperature,
            max_tokens=self._consolidation_max_tokens,
        )

        parsed = self._parse_consolidation(response.text)
        if parsed:
            for mem in parsed:
                await self.memory.store(Memory(
                    id=str(uuid.uuid4()),
                    content=mem["recuerdo"],
                    age_stage=stage.range,
                    emotional_tag=mem.get("emocion"),
                    importance=mem["importancia"],
                    memory_type="consolidated",
                ))
            logger.info(
                f"🧠 Consolidated {len(block)} turns → {len(parsed)} memories "
                f"(edad {stage.range})"
            )
        elif "{" in response.text:
            # The model tried to emit memories but none parsed — don't lose the block
            logger.warning("Consolidation output unparseable; storing mechanical fallback")
            await self._store_block_fallback(block)
        else:
            # Model judged the block not worth remembering ("NADA")
            logger.info(f"🧠 Consolidated {len(block)} turns → nothing memorable")

    @staticmethod
    def _parse_consolidation(raw: str) -> list[dict]:
        """
        Parse the consolidation reply: one JSON object per line
        ({"recuerdo", "emocion", "importancia"}). Garbage lines are
        skipped; importancia is clamped to [0, 1]. NDJSON (not a JSON
        array) so a leading '[' can't be mistaken for an emotion tag.
        """
        memories = []
        for line in raw.splitlines():
            line = line.strip().strip("`").strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                content = str(data.get("recuerdo", "")).strip()
                if not content:
                    continue
                importance = min(1.0, max(0.0, float(data.get("importancia", 0.5))))
                emotion = str(data.get("emocion", "")).strip() or None
                memories.append(
                    {"recuerdo": content, "emocion": emotion, "importancia": importance}
                )
            except (ValueError, TypeError):
                continue
        return memories[:3]

    async def _store_block_fallback(self, block: list[ConversationTurn]) -> None:
        """Mechanical fallback when consolidation fails: store a truncated
        first-exchange summary (today's pre-consolidation behavior) so the
        block leaves at least some trace in long-term memory."""
        if not (self.memory and block):
            return
        name = self.config.get("llm", {}).get("personality", {}).get("name", "Indira")
        user_part = next((t.content for t in block if t.role == "user"), "")[:150].strip()
        reply_part = next((t.content for t in block if t.role == "assistant"), "")[:200].strip()
        if not (user_part or reply_part):
            return
        stage = self._stage_for_timestamp(block[-1].timestamp)
        await self.memory.store(Memory(
            id=str(uuid.uuid4()),
            content=f"Mamá dijo: {user_part}. {name} respondió: {reply_part}",
            age_stage=stage.range,
            importance=0.5,
            memory_type="consolidated",
        ))

    def _stage_for_timestamp(self, timestamp: float):
        """Age stage at a given wall-clock time (falls back to current)."""
        if self.state.performance_start_time is None:
            return self.age_engine.get_current_stage(self.state)
        hours = (timestamp - self.state.performance_start_time) / 3600
        return self.age_engine.stage_for_hours(hours)

    async def _store_age_milestone(self) -> None:
        """Store an age transition as a milestone memory (in Spanish — it
        gets injected into the prompt as one of the character's recuerdos)."""
        stage = self.age_engine.get_current_stage(self.state)
        low, _, high = stage.range.partition("-")
        # No mention of hours alive: the character isn't aware of the
        # accelerated timescale, only of her age (decided 2026-08-16).
        await self._store_milestone(f"Crecí: ahora tengo entre {low} y {high} años.")

    async def _store_milestone(self, content: str) -> None:
        """Store a milestone event in memory."""
        if not self.memory:
            return

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            age_stage=self.state.current_age_stage,
            importance=1.0,
            memory_type="milestone",
        )
        await self.memory.store(memory)

    async def _stream_and_speak(self, user_input: str, console: Any) -> str:
        """
        Generate a response and deliver it: stream text to the console,
        speak it through TTS. Shared by text and voice modes.

        With tts.streaming enabled, sentences are synthesized and played
        while the LLM is still generating (overlapping pipeline). Otherwise
        the full response is synthesized after generation completes.

        Returns the full response text.
        """
        from rich.markup import escape

        stage = self.age_engine.get_current_stage(self.state)
        name = self.config.get("llm", {}).get("personality", {}).get("name", "Entity")
        use_streaming = self.config.get("llm", {}).get("streaming", True)
        tts_streaming = bool(
            self.tts and self.config.get("tts", {}).get("streaming", False)
        )
        logger.debug(f"🔧 TTS streaming: {tts_streaming} | LLM streaming: {use_streaming}")

        self.state.is_speaking = True
        # Hold log output back: the response line stays open across chunks,
        # and a log record landing mid-sentence makes the console unreadable
        defer_console_logs()
        try:
            if not use_streaming:
                # Non-streaming fallback
                full_response = await self.process_input(user_input)
                emotion = escape(self.state.current_emotion or "neutral")
                console.print(
                    f"\n🎭 [bold cyan]{name}[/bold cyan] "
                    f"[dim](age {stage.range}, {emotion})[/dim]: ",
                    end="",
                )
                # markup off: brackets in the line ("[risas]") are dialogue,
                # not Rich tags — Rich would silently swallow them
                console.print(full_response, markup=False, highlight=False)
                await self._speak_full(full_response, console)
                return full_response

            # Stream response word-by-word
            header_printed = False
            response = None

            # Real-time TTS pipeline state
            tts_sentence_buf = ""     # Accumulates clean text for TTS
            tts_queue = asyncio.Queue()  # Sentences ready for TTS
            tts_task = None           # Background task playing audio
            tts_sentence_count = 0    # Sentences accumulated in buffer
            tts_chunk_size = self.config.get("tts", {}).get("chunk_sentences", 1)
            tts_metrics = []          # Accumulate TTS logs to print at end
            # Live [emoción] tag stripping (the end-of-stream parse in the
            # provider is the authority for state; this one feeds display+TTS)
            tag_filter = EmotionTagFilter()

            async for chunk in self.process_input_streaming(user_input):
                if isinstance(chunk, LLMResponse):
                    response = chunk
                    continue

                # Print header before first chunk
                if not header_printed:
                    emotion = escape(self.state.current_emotion or "neutral")
                    console.print(
                        f"\n🎭 [bold cyan]{name}[/bold cyan] "
                        f"[dim](age {stage.range}, {emotion})[/dim]: ",
                        end="",
                    )
                    header_printed = True

                clean = tag_filter.feed(chunk)
                if clean:
                    # markup off: brackets in the line ("[risas]") are dialogue,
                    # not Rich tags — Rich would silently swallow them
                    console.print(clean, end="", markup=False, highlight=False)
                    tts_sentence_buf += clean

                # Check for sentence boundary → count and queue for TTS
                if tts_streaming and tts_sentence_buf.rstrip().endswith((".", "!", "?")):
                    tts_sentence_count += 1
                    if tts_sentence_count >= tts_chunk_size:
                        # Sanitized: emojis/tags/actions derail the TTS
                        # (a lone emoji synthesizes as CJK babble)
                        sentence = sanitize_for_tts(tts_sentence_buf)
                        tts_sentence_buf = ""
                        tts_sentence_count = 0
                        if sentence:
                            await tts_queue.put(sentence)
                            # Start TTS consumer if not running
                            if tts_task is None or tts_task.done():
                                voice_profile = self._get_voice_profile()
                                emotion_for_tts = tag_filter.emotion or self.state.current_emotion
                                tts_task = asyncio.create_task(
                                    self._tts_consumer(
                                        tts_queue, voice_profile, emotion_for_tts, tts_metrics
                                    )
                                )

            # Release any text still held by the tag filter — short untagged
            # replies would otherwise never reach the console or TTS
            tail = tag_filter.flush()
            if tail:
                console.print(tail, end="", markup=False, highlight=False)
                tts_sentence_buf += tail

            # Flush remaining text to TTS (sanitized — see above)
            if tts_streaming and sanitize_for_tts(tts_sentence_buf):
                await tts_queue.put(sanitize_for_tts(tts_sentence_buf))
                if tts_task is None or tts_task.done():
                    voice_profile = self._get_voice_profile()
                    emotion_for_tts = tag_filter.emotion or self.state.current_emotion
                    tts_task = asyncio.create_task(
                        self._tts_consumer(
                            tts_queue, voice_profile, emotion_for_tts, tts_metrics
                        )
                    )

            # Newline after streaming completes
            console.print()

            # Print stats below the response
            if response:
                ttft = response.first_token_time_ms
                total = response.generation_time_ms
                tokens = response.tokens_generated
                tps = (tokens / (total / 1000)) if total > 0 else 0
                console.print(
                    f"[dim]   ⚡ first token: {ttft:.0f}ms | "
                    f"total: {total:.0f}ms | "
                    f"{tokens} tokens ({tps:.1f} tok/s) | "
                    f"emotion: {response.emotion or 'none'}[/dim]"
                )

            full_response = response.text if response else ""

            # Wait for TTS to finish all queued sentences
            if tts_streaming and tts_task and not tts_task.done():
                await tts_queue.put(None)  # Sentinel to stop consumer
                await tts_task

            # Print TTS metrics at the end
            if tts_streaming and tts_metrics:
                for m in tts_metrics:
                    console.print(f"[dim]   🔉 {m}[/dim]")

            # Non-streaming TTS: speak the full response at once
            if not tts_streaming:
                await self._speak_full(full_response, console)

            return full_response
        finally:
            self.state.is_speaking = False
            flush_console_logs()

    async def _speak_full(self, text_to_speak: str, console: Any) -> None:
        """Synthesize a whole response at once and play it."""
        # Sanitized: emojis/tags/actions derail the TTS (CJK babble)
        text_to_speak = sanitize_for_tts(text_to_speak) if text_to_speak else ""
        if not (self.tts and text_to_speak):
            return

        voice_profile = self._get_voice_profile()
        emotion_for_tts = self.state.current_emotion
        try:
            tts_result = await self.tts.synthesize(
                text_to_speak, voice_profile, emotion=emotion_for_tts
            )
            console.print(
                f"[dim]   🔉 TTS [{len(text_to_speak)} chars]: "
                f"{tts_result.duration_seconds:.1f}s audio "
                f"generated in {tts_result.generation_time_ms:.0f}ms "
                f"(emotion: {emotion_for_tts or 'none'})[/dim]"
            )
            playback = await WavPlayback.start(tts_result.audio_data)
            await playback.wait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
            console.print(f"[dim]   🔇 TTS error: {e}[/dim]")

    async def run_text_mode(self) -> None:
        """
        Interactive text-based conversation mode.

        Great for testing without microphone setup.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        console.print(Panel.fit(
            "[bold magenta]🎭 AI Actor — Text Mode[/bold magenta]\n"
            "[dim]Type your messages. Commands: /age, /status, /memory, /lobotomy, /quit[/dim]",
            border_style="magenta",
        ))

        if self._start_or_resume_performance():
            console.print(
                f"[yellow]⏮️  Resumed at hour {self.state.hours_elapsed:.1f} "
                f"(age {self.state.current_age_stage}). "
                f"Use --fresh to start over.[/yellow]"
            )

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n🎤 You: ")
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                await self._handle_command(user_input, console)
                continue

            # Process input
            try:
                await self._stream_and_speak(user_input, console)
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                logger.exception("Error processing input")

    async def run_voice_mode(self) -> None:
        """
        Interactive voice-based conversation mode.

        Captures microphone audio, transcribes with mlx-whisper,
        processes through LLM, and plays TTS response.
        """
        import sounddevice as sd
        import numpy as np
        from collections import deque
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        if not self.stt:
            console.print("[red]STT provider not initialized. Cannot use voice mode.[/red]")
            return
        if not self.tts:
            console.print("[red]TTS provider not initialized. Cannot use voice mode.[/red]")
            return

        # Audio config
        audio_config = self.config.get("audio", {})
        sample_rate = audio_config.get("sample_rate", 16000)
        channels = audio_config.get("channels", 1)

        # VAD config
        vad_config = self.config.get("stt", {}).get("vad", {})
        energy_threshold = vad_config.get("energy_threshold", 0.02)
        silence_ms = vad_config.get("silence_duration_ms", 800)
        silence_frames = int(silence_ms / 1000 * sample_rate)
        preroll_ms = vad_config.get("preroll_ms", 300)

        console.print(Panel.fit(
            "[bold magenta]🎭 AI Actor — Voice Mode[/bold magenta]\n"
            "[dim]Speak into your microphone. Press Ctrl+C to quit.[/dim]\n"
            f"[dim]VAD threshold: {energy_threshold} | Silence: {silence_ms}ms[/dim]",
            border_style="magenta",
        ))

        if self._start_or_resume_performance():
            console.print(
                f"[yellow]⏮️  Resumed at hour {self.state.hours_elapsed:.1f} "
                f"(age {self.state.current_age_stage}). "
                f"Use --fresh to start over.[/yellow]"
            )

        # Audio buffer shared between callback and main loop
        audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio chunk."""
            if status:
                logger.warning(f"Audio input status: {status}")
            # Copy data and put in queue (callback runs in a separate thread)
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

        # Open mic stream
        block_duration_ms = 100  # Process audio in 100ms blocks
        block_size = int(sample_rate * block_duration_ms / 1000)
        preroll_maxlen = max(1, round(preroll_ms / block_duration_ms))

        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=block_size,
                callback=audio_callback,
            )
            stream.start()
            logger.info(f"🎤 Microphone opened (rate={sample_rate}, block={block_size})")
        except Exception as e:
            console.print(f"[red]Failed to open microphone: {e}[/red]")
            console.print("[dim]Check your audio input device in System Settings.[/dim]")
            return

        console.print("\n[green]🎤 Listening...[/green]", end="")

        try:
            while True:
                # ── Phase 1: Wait for speech ──
                speech_buffer: list[np.ndarray] = []
                # Rolling ~300ms of pre-speech audio: speech onsets fade in
                # below the energy threshold, so without this the first
                # syllable gets clipped and Whisper mis-hears the line start.
                preroll: deque[np.ndarray] = deque(maxlen=preroll_maxlen)
                is_speaking = False
                silence_counter = 0

                while True:
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue

                    if chunk is None:
                        break

                    energy = np.sqrt(np.mean(chunk ** 2))

                    if energy > energy_threshold:
                        if not is_speaking:
                            is_speaking = True
                            silence_counter = 0
                            preroll_samples = sum(c.size for c in preroll)
                            speech_buffer.extend(preroll)
                            preroll.clear()
                            console.print("\r[yellow]🎤 Hearing you...[/yellow]   ", end="")
                        speech_buffer.append(chunk)
                        silence_counter = 0
                    elif not is_speaking:
                        preroll.append(chunk)
                    elif is_speaking:
                        # Still recording but silence detected
                        speech_buffer.append(chunk)
                        silence_counter += len(chunk)

                        if silence_counter >= silence_frames:
                            # Enough silence — speech ended
                            break

                if not speech_buffer:
                    continue

                # ── Anti-hallucination checks ──
                # Checks run on the speech portion only — the quiet pre-roll
                # would otherwise pad blips past the duration check and drag
                # the average energy down.
                audio_data = np.concatenate(speech_buffer).flatten()
                speech_only = audio_data[preroll_samples:]
                duration_s = len(speech_only) / sample_rate

                # 1. Minimum speech duration — ignore noise blips
                if duration_s < 0.5:
                    console.print("\r[green]🎤 Listening...[/green]", end="")
                    continue

                # 2. Average energy of the speech portion — reject near-silent audio
                avg_energy = np.sqrt(np.mean(speech_only ** 2))
                if avg_energy < energy_threshold * 0.7:
                    logger.debug(f"Audio too quiet (energy={avg_energy:.4f}), skipping")
                    console.print("\r[green]🎤 Listening...[/green]", end="")
                    continue

                # ── Phase 2: Transcribe ──
                console.print("\r[cyan]💭 Transcribing...[/cyan]   ", end="")
                audio_bytes = audio_data.astype(np.float32).tobytes()

                try:
                    transcription = await self.stt.transcribe(audio_bytes, sample_rate)
                except Exception as e:
                    logger.error(f"STT error: {e}")
                    console.print(f"\r[red]STT error: {e}[/red]")
                    console.print("\n[green]🎤 Listening...[/green]", end="")
                    continue

                text = transcription.text.strip()

                # 3. Filter known Whisper hallucination patterns
                _hallucination_patterns = [
                    "thank you", "thanks for watching", "subscribe",
                    "subtítulos", "amara.org", "gracias por ver",
                    "música", "aplausos", "risas",
                ]
                text_lower = text.lower()
                if (
                    not text
                    or len(text) < 2
                    or any(p in text_lower for p in _hallucination_patterns)
                    or text_lower == text_lower[0] * len(text_lower)  # repeated single char
                ):
                    if text:
                        logger.debug(f"Filtered hallucination: '{text}'")
                    console.print("\r[green]🎤 Listening...[/green]", end="")
                    continue

                # Show what was heard
                console.print(f"\r🎤 You ({transcription.language}): {text}")

                # ── Phase 3: Generate and speak (mic paused) ──
                # The mic pauses for the WHOLE generate+speak phase: with
                # streaming TTS, playback starts while the LLM is still
                # generating, and the mic must not hear the speakers.
                stream.stop()
                try:
                    await self._stream_and_speak(text, console)
                except Exception as e:
                    logger.error(f"Response error: {e}")
                    console.print(f"\r[red]Error: {e}[/red]")
                finally:
                    # Drain audio captured before the mic was paused, then resume
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    stream.start()

                console.print("\n[green]🎤 Listening...[/green]", end="")

        except KeyboardInterrupt:
            console.print("\n")
        finally:
            stream.stop()
            stream.close()
            logger.info("🎤 Microphone closed")

    async def _tts_consumer(
        self,
        queue: asyncio.Queue,
        voice_profile: VoiceProfile | None,
        emotion: str | None,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Background task that consumes sentences from the queue and plays audio.

        Overlaps synthesis and playback: generates next chunk while current plays.
        """
        if not self.tts:
            return

        playback: WavPlayback | None = None
        try:
            while True:
                sentence = await queue.get()
                if sentence is None:  # Sentinel: stop
                    break

                # Generate audio
                try:
                    result = await self.tts.synthesize(
                        sentence, voice_profile, emotion
                    )
                    if metrics is not None:
                        metrics.append(
                            f"TTS [{len(sentence)} chars]: "
                            f"{result.duration_seconds:.1f}s audio generated "
                            f"in {result.generation_time_ms:.0f}ms"
                        )
                except Exception as e:
                    logger.warning(f"TTS consumer error: {e}")
                    continue

                # Wait for the previous sentence to finish, then start this
                # one non-blocking (each playback cleans up its own file)
                if playback is not None:
                    await playback.wait()
                playback = await WavPlayback.start(result.audio_data)

            # Wait for last audio to finish
            if playback is not None:
                await playback.wait()

        finally:
            # Cancellation path: don't leak an in-flight playback's temp file
            if playback is not None:
                playback.cleanup()

    async def _handle_command(self, command: str, console: Any) -> None:
        """Handle special commands in text mode."""
        from rich.panel import Panel

        if command == "/quit":
            raise KeyboardInterrupt

        elif command == "/status":
            stage = self.age_engine.get_current_stage(self.state)
            mem_count = await self.memory.count() if self.memory else 0
            console.print(Panel(
                f"⏱️  Hours elapsed: {self.state.hours_elapsed:.2f}\n"
                f"🎂 Age stage: {stage.range}\n"
                f"😊 Emotion: {self.state.current_emotion}\n"
                f"🧠 Memories: {mem_count}\n"
                f"💬 Conversation turns: {len(self.state.conversation_history)}",
                title="Status",
                border_style="cyan",
            ))

        elif command.startswith("/age"):
            # Set literal human age: /age 25 -> jumps to the 25-30 age stage
            parts = command.split()
            if len(parts) < 2:
                console.print("[yellow]Usage: /age <human_age> — e.g. /age 25[/yellow]")
                return
            try:
                target_age = float(parts[1])
                target_hour = None
                
                for stage in self.age_engine.stages:
                    range_parts = stage.range.split("-")
                    if len(range_parts) == 2:
                        try:
                            start_age = float(range_parts[0])
                            end_age = float(range_parts[1])
                            if start_age <= target_age <= end_age:
                                target_hour = stage.start_hour
                                break
                        except ValueError:
                            pass
                
                if target_hour is None:
                    # If target age is outside all ranges, snap to the closest
                    if target_age < 10:
                        target_hour = self.age_engine.stages[0].start_hour
                    else:
                        target_hour = self.age_engine.stages[-1].start_hour
                
                if self.state.performance_start_time:
                    self.state.performance_start_time = time.time() - (target_hour * 3600.0)
                    self.age_engine.update_state(self.state)
                    self._persist_performance_state()
                    stage = self.age_engine.get_current_stage(self.state)
                    console.print(f"⏩ Set age to {target_age}. Mapped to stage: {stage.range}")
            except ValueError:
                console.print("[yellow]Usage: /age <human_age> — e.g. /age 25[/yellow]")

        elif command == "/memory":
            if self.memory:
                recent = await self.memory.get_recent(5)
                if recent:
                    console.print("[bold]Recent Memories:[/bold]")
                    for mem in recent:
                        console.print(f"  [{mem.age_stage}] {mem.content[:80]}")
                else:
                    console.print("[dim]No memories yet[/dim]")

        elif command == "/lobotomy":
            if self.memory:
                await self.memory.clear()
                console.print("[bold red]🧠 LOBOTOMY COMPLETE: All memories have been erased.[/bold red]")
            else:
                console.print("[dim]No memory provider active to lobotomize.[/dim]")

        elif command == "/help":
            console.print(
                "/status   — Show system status\n"
                "/age N    — Set character to N years old\n"
                "/memory   — Show recent memories\n"
                "/lobotomy — Erase all stored memories\n"
                "/quit     — Exit"
            )

        else:
            console.print(f"[dim]Unknown command. Type /help[/dim]")
