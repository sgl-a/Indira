from __future__ import annotations

"""
Orchestrator.

The central coordinator that connects all providers and manages
the AI Actor's conversation loop. Handles:
- Audio input → STT → LLM → TTS → Audio output
- Memory storage and retrieval
- Age progression
- Proactive conversation initiation
"""

import asyncio
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

from src.core.config import get_config
from src.core.state import ActorState, ConversationTurn
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
        self._proactive_task: asyncio.Task | None = None

    async def setup(self) -> None:
        """Initialize all providers."""
        logger.info("🎭 Initializing AI Actor system...")

        # Initialize providers in parallel
        self.llm = await create_llm_provider(self.config)
        self.tts = await create_tts_provider(self.config)
        self.memory = await create_memory_provider(self.config)

        # STT is optional for text-only mode
        try:
            self.stt = await create_stt_provider(self.config)
        except Exception as e:
            logger.warning(f"STT initialization failed: {e}. Running in text-only mode.")

        logger.info("✅ All providers initialized")

    async def shutdown(self) -> None:
        """Clean shutdown of all providers."""
        logger.info("Shutting down AI Actor system...")
        self._running = False

        if self._proactive_task:
            self._proactive_task.cancel()

        for provider in [self.stt, self.llm, self.tts, self.memory]:
            if provider:
                try:
                    await provider.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {provider.__class__.__name__}: {e}")

        logger.info("👋 AI Actor system shut down")

    async def process_input(self, user_text: str, speaker_id: str | None = None) -> str:
        """
        Process text input and generate a response.

        This is the core conversation loop:
        1. Store user input
        2. Retrieve relevant memories
        3. Build context (personality + memories + history)
        4. Generate response via LLM
        5. Store significant memories
        6. Synthesize speech (if TTS available)

        Args:
            user_text: What the user (actress) said
            speaker_id: Optional speaker identifier

        Returns:
            The AI's spoken response text
        """
        if not self.llm:
            raise RuntimeError("LLM provider not initialized")

        # Update age progression
        age_changed = self.age_engine.update_state(self.state)
        if age_changed:
            # Store age transition as milestone memory
            stage = self.age_engine.get_current_stage(self.state)
            await self._store_milestone(
                f"Aged into {stage.range} years old after {self.state.hours_elapsed:.1f} hours"
            )

        # 1. Record user input
        self.state.add_turn("user", user_text, speaker_id=speaker_id)

        # 2. Retrieve relevant memories
        memories_context = await self._build_memory_context(user_text)

        # 3. Build system prompt
        system_prompt = self.age_engine.build_personality_prompt(self.state, self.config)

        if memories_context:
            system_prompt += f"\n\n## Tus recuerdos\n{memories_context}"

        # 4. Get conversation history
        messages = self.state.get_recent_messages(limit=20)

        # 5. Generate response
        llm_config = self.config.get("llm", {})
        response = await self.llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=llm_config.get("temperature", 0.8),
            max_tokens=llm_config.get("max_tokens", 512),
        )

        # 6. Update state with response
        self.state.add_turn("assistant", response.text, emotion=response.emotion)
        if response.emotion:
            self.state.current_emotion = response.emotion

        # 7. Store in memory if significant
        if response.should_store_memory:
            await self._store_interaction(user_text, response)

        # Log performance
        logger.debug(
            f"💬 Response ({response.generation_time_ms:.0f}ms, "
            f"{response.tokens_generated} tokens, "
            f"emotion={response.emotion}): {response.text[:80]}..."
        )

        return response.text

    async def process_input_streaming(
        self, user_text: str, speaker_id: str | None = None
    ) -> AsyncIterator[str | LLMResponse]:
        """
        Process text input and stream the response word-by-word.

        Yields str chunks as they arrive, then a final LLMResponse.
        The caller should display chunks in real-time, then use the
        LLMResponse for metadata (emotion, timing, memory storage).
        """
        if not self.llm:
            raise RuntimeError("LLM provider not initialized")

        # Update age progression
        age_changed = self.age_engine.update_state(self.state)
        if age_changed:
            stage = self.age_engine.get_current_stage(self.state)
            await self._store_milestone(
                f"Aged into {stage.range} years old after {self.state.hours_elapsed:.1f} hours"
            )

        # 1. Record user input
        self.state.add_turn("user", user_text, speaker_id=speaker_id)

        # 2. Retrieve relevant memories
        memories_context = await self._build_memory_context(user_text)

        # 3. Build system prompt
        system_prompt = self.age_engine.build_personality_prompt(self.state, self.config)
        if memories_context:
            system_prompt += f"\n\n## Tus recuerdos\n{memories_context}"

        # 4. Get conversation history
        messages = self.state.get_recent_messages(limit=20)

        # 5. Stream response
        llm_config = self.config.get("llm", {})
        response = None

        if hasattr(self.llm, "stream_generate_with_metadata"):
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
        else:
            # Fallback: non-streaming
            response = await self.llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                temperature=llm_config.get("temperature", 0.8),
                max_tokens=llm_config.get("max_tokens", 512),
            )
            yield response.text

        # 6. Post-streaming: update state and store memory
        if response:
            self.state.add_turn("assistant", response.text, emotion=response.emotion)
            if response.emotion:
                self.state.current_emotion = response.emotion
            if response.should_store_memory:
                await self._store_interaction(user_text, response)


        # Yield final response for caller to use if needed
        if response:
            yield response

    async def process_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str | None:
        """
        Process audio input through STT → LLM → TTS pipeline.

        Returns the response text, or None if no speech was detected.
        """
        if not self.stt:
            raise RuntimeError("STT provider not initialized")

        # 1. Transcribe audio
        transcription = await self.stt.transcribe(audio_data, sample_rate)

        if not transcription.text.strip():
            return None

        logger.info(f"🎤 Heard ({transcription.language}): {transcription.text}")

        # 2. Process through conversation
        response_text = await self.process_input(transcription.text)

        # 3. Synthesize speech
        if self.tts:
            voice_profile = self._get_voice_profile()
            tts_result = await self.tts.synthesize(
                text=response_text,
                voice_profile=voice_profile,
                emotion=self.state.current_emotion,
            )
            logger.info(f"🔊 TTS generated ({tts_result.generation_time_ms:.0f}ms)")
            # Audio playback would be handled by the output module

        return response_text

    def _get_voice_profile(self) -> VoiceProfile:
        """Get the voice profile for the current age stage."""
        stage = self.age_engine.get_current_stage(self.state)
        return VoiceProfile(
            id=f"voice_{stage.range}",
            name=f"Voice {stage.range}",
            age_stage=stage.range,
            reference_audio_path=stage.voice_profile_path or "",
        )

    async def _build_memory_context(self, query: str) -> str:
        """Build memory context string for the LLM prompt."""
        if not self.memory:
            return ""

        # Get relevant memories
        relevant = await self.memory.search(query, limit=5)
        recent = await self.memory.get_recent(limit=3)

        # Deduplicate
        seen_ids = set()
        all_memories = []
        for mem in relevant + recent:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                all_memories.append(mem)

        if not all_memories:
            return ""

        parts = ["Cosas que recordás:"]
        for mem in all_memories:
            emotion_str = f" (sintiendo {mem.emotional_tag})" if mem.emotional_tag else ""
            parts.append(f"- [Edad {mem.age_stage}]{emotion_str}: {mem.content}")

        context = "\n".join(parts)
        logger.debug(f"📝 Memory context ({len(all_memories)} memories injected into prompt)")
        return context

    async def _store_interaction(self, user_input: str, response: LLMResponse) -> None:
        """Store a significant interaction in memory."""
        if not self.memory:
            return

        # Build a useful summary from both sides of the conversation.
        # The LLM rarely sets memory_summary, so we build one ourselves.
        if response.memory_summary:
            summary = response.memory_summary
        else:
            # Include both what was said and what was responded —
            # this is what makes semantic search actually work.
            user_part = user_input[:150].strip()
            response_part = response.text[:200].strip()
            summary = f"Mamá dijo: {user_part}. Indira respondió: {response_part}"

        memory = Memory(
            id=str(uuid.uuid4()),
            content=summary,
            age_stage=self.state.current_age_stage,
            emotional_tag=response.emotion,
            importance=0.5,
            memory_type="interaction",
        )
        await self.memory.store(memory)

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

        self.state.start_performance()

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
                self.state.is_speaking = True
                stage = self.age_engine.get_current_stage(self.state)
                use_streaming = self.config.get("llm", {}).get("streaming", True)
                personality = self.config.get("llm", {}).get("personality", {})
                name = personality.get("name", "Entity")
                tts_streaming = (
                    self.tts
                    and hasattr(self.tts, "speak_directly")
                    and self.config.get("tts", {}).get("streaming", False)
                )
                logger.info(f"🔧 TTS streaming: {tts_streaming} | LLM streaming: {use_streaming}")

                if use_streaming:
                    # Stream response word-by-word
                    header_printed = False
                    response = None

                    # Real-time TTS pipeline state
                    tts_sentence_buf = ""     # Accumulates clean text for TTS
                    tts_queue = asyncio.Queue()  # Sentences ready for TTS
                    tts_task = None           # Background task playing audio
                    stream_buffer = ""        # Buffer to strip [emotion] tag
                    tag_stripped = False
                    tts_sentence_count = 0    # Sentences accumulated in buffer
                    tts_chunk_size = self.config.get("tts", {}).get("chunk_sentences", 1)
                    tts_metrics = []          # Accumulate TTS logs to print at end
                    current_streaming_emotion = self.state.current_emotion  # Fallback

                    async for chunk in self.process_input_streaming(user_input):
                        if isinstance(chunk, LLMResponse):
                            response = chunk
                            continue

                        # Print header before first chunk
                        if not header_printed:
                            from rich.markup import escape
                            emotion = escape(self.state.current_emotion or "neutral")
                            console.print(
                                f"\n🎭 [bold cyan]{name}[/bold cyan] "
                                f"[dim](age {stage.range}, {emotion})[/dim]: ",
                                end="",
                            )
                            header_printed = True

                        # Buffer initial chunks to strip [emotion] tag
                        if not tag_stripped:
                            stream_buffer += chunk
                            bracket_end = stream_buffer.find("]")
                            if bracket_end >= 0 and stream_buffer.lstrip().startswith("["):
                                bracket_start = stream_buffer.find("[")
                                current_streaming_emotion = stream_buffer[bracket_start + 1:bracket_end].strip()
                                clean = stream_buffer[bracket_end + 1:].lstrip()
                                if clean:
                                    console.print(clean, end="", highlight=False)
                                    tts_sentence_buf += clean
                                tag_stripped = True
                            elif len(stream_buffer) > 50 or (not stream_buffer.lstrip().startswith("[") and len(stream_buffer) > 5):
                                console.print(stream_buffer, end="", highlight=False)
                                tts_sentence_buf += stream_buffer
                                tag_stripped = True
                        else:
                            # Print each chunk as it arrives
                            console.print(chunk, end="", highlight=False)
                            tts_sentence_buf += chunk

                        # Check for sentence boundary → count and queue for TTS
                        if tts_streaming and tag_stripped and tts_sentence_buf.rstrip().endswith((".", "!", "?")):
                            tts_sentence_count += 1
                            if tts_sentence_count >= tts_chunk_size:
                                sentence = tts_sentence_buf.strip()
                                tts_sentence_buf = ""
                                tts_sentence_count = 0
                                if sentence:
                                    await tts_queue.put(sentence)
                                    # Start TTS consumer if not running
                                    if tts_task is None or tts_task.done():
                                        voice_profile = self._get_voice_profile()
                                        emotion_for_tts = current_streaming_emotion
                                        tts_task = asyncio.create_task(
                                            self._tts_consumer(
                                                tts_queue, voice_profile, emotion_for_tts, tts_metrics
                                            )
                                        )

                    # Flush remaining text to TTS
                    if tts_streaming and tts_sentence_buf.strip():
                        await tts_queue.put(tts_sentence_buf.strip())
                        if tts_task is None or tts_task.done():
                            voice_profile = self._get_voice_profile()
                            tts_task = asyncio.create_task(
                                self._tts_consumer(
                                    tts_queue, voice_profile, current_streaming_emotion, tts_metrics
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
                else:
                    # Non-streaming fallback
                    full_response = await self.process_input(user_input)
                    from rich.markup import escape
                    emotion = escape(self.state.current_emotion or "neutral")
                    console.print(
                        f"\n🎭 [bold cyan]{name}[/bold cyan] "
                        f"[dim](age {stage.range}, {emotion})[/dim]: {full_response}"
                    )

                self.state.is_speaking = False

                # Speak with TTS (non-streaming path: full response at once)
                if not tts_streaming and self.tts and full_response:
                    import tempfile
                    from pathlib import Path
                    voice_profile = self._get_voice_profile()
                    emotion_for_tts = self.state.current_emotion
                    tts_result = await self.tts.synthesize(
                        full_response, voice_profile, emotion=emotion_for_tts
                    )
                    console.print(
                        f"[dim]   🔉 TTS [{len(full_response)} chars]: "
                        f"{tts_result.duration_seconds:.1f}s audio "
                        f"generated in {tts_result.generation_time_ms:.0f}ms "
                        f"(emotion: {emotion_for_tts or 'none'})[/dim]"
                    )
                    # Play the already-synthesized audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(tts_result.audio_data)
                        temp_path = f.name
                    try:
                        process = await asyncio.create_subprocess_exec(
                            "afplay", temp_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await process.wait()
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

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
        import tempfile
        from pathlib import Path
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

        personality = self.config.get("llm", {}).get("personality", {})
        name = personality.get("name", "Entity")

        console.print(Panel.fit(
            "[bold magenta]🎭 AI Actor — Voice Mode[/bold magenta]\n"
            "[dim]Speak into your microphone. Press Ctrl+C to quit.[/dim]\n"
            f"[dim]VAD threshold: {energy_threshold} | Silence: {silence_ms}ms[/dim]",
            border_style="magenta",
        ))

        self.state.start_performance()

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
                            console.print("\r[yellow]🎤 Hearing you...[/yellow]   ", end="")
                        speech_buffer.append(chunk)
                        silence_counter = 0
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
                audio_data = np.concatenate(speech_buffer).flatten()
                duration_s = len(audio_data) / sample_rate

                # 1. Minimum speech duration — ignore noise blips
                if duration_s < 0.5:
                    console.print("\r[green]🎤 Listening...[/green]", end="")
                    continue

                # 2. Average energy of the full buffer — reject near-silent audio
                avg_energy = np.sqrt(np.mean(audio_data ** 2))
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
                stage = self.age_engine.get_current_stage(self.state)
                console.print(f"\r🎤 You ({transcription.language}): {text}")

                # ── Phase 3: Process through LLM ──
                try:
                    full_response = await self.process_input(text)
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    console.print(f"\r[red]LLM error: {e}[/red]")
                    console.print("\n[green]🎤 Listening...[/green]", end="")
                    continue

                # Show response
                stage = self.age_engine.get_current_stage(self.state)
                from rich.markup import escape
                emotion = escape(self.state.current_emotion or "neutral")
                console.print(
                    f"\r🎭 [bold cyan]{name}[/bold cyan] "
                    f"[dim](age {stage.range}, {emotion})[/dim]: {full_response}"
                )

                # ── Phase 4: Speak response (mic paused) ──
                if self.tts and full_response:
                    # Pause mic so TTS output doesn't get picked up as input
                    stream.stop()

                    voice_profile = self._get_voice_profile()
                    emotion_for_tts = self.state.current_emotion
                    try:
                        tts_result = await self.tts.synthesize(
                            full_response, voice_profile, emotion=emotion_for_tts
                        )
                        console.print(
                            f"[dim]   🔉 TTS [{len(full_response)} chars]: "
                            f"{tts_result.duration_seconds:.1f}s audio "
                            f"in {tts_result.generation_time_ms:.0f}ms "
                            f"(emotion: {emotion_for_tts or 'none'})[/dim]"
                        )

                        # Play audio
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            f.write(tts_result.audio_data)
                            temp_path = f.name
                        try:
                            process = await asyncio.create_subprocess_exec(
                                "afplay", temp_path,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            await process.wait()
                        finally:
                            Path(temp_path).unlink(missing_ok=True)

                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        console.print(f"[dim]   🔇 TTS error: {e}[/dim]")

                    # Drain any queued audio captured before mic was paused
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    # Resume mic
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
        import tempfile
        from pathlib import Path

        if not self.tts:
            return

        play_process = None
        temp_files = []

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

                # Wait for previous audio to finish
                if play_process is not None:
                    await play_process.wait()

                # Start playing (non-blocking)
                tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tf.write(result.audio_data)
                tf.close()
                temp_files.append(tf.name)

                play_process = await asyncio.create_subprocess_exec(
                    "afplay", tf.name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            # Wait for last audio to finish
            if play_process is not None:
                await play_process.wait()

        finally:
            for f in temp_files:
                Path(f).unlink(missing_ok=True)

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
                    import time
                    self.state.performance_start_time = time.time() - (target_hour * 3600.0)
                    self.age_engine.update_state(self.state)
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
