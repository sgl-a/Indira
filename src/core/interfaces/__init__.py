"""
Core interfaces for the Indira system.

All providers must implement these abstract interfaces.
This ensures any component can be swapped without changing the rest of the system.
"""

from src.core.interfaces.stt import STTProvider, TranscriptionResult
from src.core.interfaces.llm import LLMProvider, LLMResponse
from src.core.interfaces.tts import TTSProvider, VoiceProfile
from src.core.interfaces.memory import MemoryProvider, Memory

__all__ = [
    "STTProvider",
    "TranscriptionResult",
    "LLMProvider",
    "LLMResponse",
    "TTSProvider",
    "VoiceProfile",
    "MemoryProvider",
    "Memory",
]
