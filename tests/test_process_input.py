"""
Tests for the unified conversation turn: process_input is a thin consumer
of process_input_streaming (single turn implementation), and a provider
without real token streaming works through the LLMProvider interface's
built-in generate() fallback.
"""

from src.core.interfaces.llm import LLMProvider, LLMResponse
from src.core.orchestrator import Orchestrator


class MinimalLLM(LLMProvider):
    """Implements only the abstract methods — no streaming override, so the
    ABC's default stream_generate_with_metadata (generate fallback) is used."""

    async def initialize(self, config: dict) -> None:
        pass

    async def generate(self, system_prompt, messages, temperature=0.7, max_tokens=512):
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        return LLMResponse(text="Hola má, ¿cómo estás?", emotion="contenta")

    async def get_model_info(self) -> dict:
        return {}


def make_orchestrator(tmp_path) -> Orchestrator:
    orch = Orchestrator({
        "system": {
            "state_file": str(tmp_path / "performance_state.json"),
            "transcript_file": str(tmp_path / "transcript.jsonl"),
        },
    })
    orch.state.start_performance()
    orch.llm = MinimalLLM()
    return orch


async def test_process_input_runs_full_turn(tmp_path):
    orch = make_orchestrator(tmp_path)

    text = await orch.process_input("hola hija")

    assert text == "Hola má, ¿cómo estás?"
    # Full turn lifecycle happened in the (single) streaming implementation:
    assert [t.role for t in orch.state.conversation_history] == ["user", "assistant"]
    assert orch.state.current_emotion == "contenta"
    assert len((tmp_path / "transcript.jsonl").read_text().splitlines()) == 2


async def test_streaming_path_yields_text_then_response(tmp_path):
    orch = make_orchestrator(tmp_path)

    chunks = [c async for c in orch.process_input_streaming("hola hija")]

    # ABC fallback: one text chunk, then the LLMResponse
    assert chunks[0] == "Hola má, ¿cómo estás?"
    assert isinstance(chunks[-1], LLMResponse)
    assert chunks[-1].emotion == "contenta"


async def test_memory_envelope_reaches_llm_only_on_newest_message(tmp_path):
    orch = make_orchestrator(tmp_path)

    class FakeMemory:
        async def search(self, query, limit=5):
            from src.core.interfaces.memory import Memory
            return [Memory(id="1", content="Mamá me contó del río.", age_stage="10-15")]

    orch.memory = FakeMemory()
    await orch.process_input("primer mensaje")
    await orch.process_input("segundo mensaje")

    # The envelope wraps the outgoing newest message…
    assert "[Contexto" in orch.llm.last_messages[-1]["content"]
    assert "segundo mensaje" in orch.llm.last_messages[-1]["content"]
    # …but replayed history keeps the clean stored text (prefix-stable)
    assert orch.llm.last_messages[0]["content"] == "primer mensaje"
