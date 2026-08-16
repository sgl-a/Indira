"""Tests for incremental <think> block stripping in the token stream."""

from src.providers.llm.ollama_provider import OllamaLLMProvider, _ThinkTagFilter


def run_filter(chunks: list[str]) -> str:
    f = _ThinkTagFilter()
    out = "".join(f.feed(c) for c in chunks)
    return out + f.flush()


def test_no_tags_passes_through():
    assert run_filter(["Hola ", "mamá"]) == "Hola mamá"


def test_think_block_in_single_chunk():
    assert run_filter(["<think>razonando...</think>Hola"]) == "Hola"


def test_think_block_split_across_chunks():
    assert run_filter(["<thi", "nk>razon", "ando</th", "ink>Hola"]) == "Hola"


def test_char_by_char_streaming():
    text = "<think>x</think>Hola mamá"
    assert run_filter(list(text)) == "Hola mamá"


def test_text_before_and_after_block():
    assert run_filter(["Hola<think>x</think> mamá"]) == "Hola mamá"


def test_unclosed_think_is_dropped():
    # Stream ended mid-reasoning: nothing inside the block should leak
    assert run_filter(["Hola<think>razonando sin cerrar"]) == "Hola"


def test_multiple_blocks():
    assert run_filter(["a<think>1</think>b<think>2</think>c"]) == "abc"


def test_angle_bracket_that_is_not_a_tag():
    assert run_filter(["2 < 3 y 5 > 4"]) == "2 < 3 y 5 > 4"


def test_partial_open_lookalike_is_flushed():
    # "<th" at end of stream is not a tag — must not be swallowed
    assert run_filter(["Hola <th"]) == "Hola <th"


def test_matches_batch_strip():
    # The incremental filter must agree with the batch regex version
    raw = "antes<think>uno\ndos</think> después <think>tres</think>fin"
    incremental = run_filter(list(raw))
    batch = OllamaLLMProvider._strip_thinking(raw)
    assert incremental.strip() == batch
