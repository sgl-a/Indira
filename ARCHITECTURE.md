# AI Actor - System Architecture

> **Last Updated**: 2026-02-01  
> **Status**: Draft / Planning

---

## Overview

This document describes the system architecture for a 72-hour AI theatrical performance. The AI "being" perceives the world through cameras and microphones, processes input through an LLM brain, and expresses itself through synthesized voice and animated face.

---

## High-Level Data Flow

```mermaid
flowchart TB
    subgraph WORLD ["🌍 THE WORLD"]
        ACTRESS["👩‍🎭 Actress"]
        AUDIENCE["👥 Audience"]
        SMOKE["💨 Smoke Screen"]
        SPEAKERS["🔊 Speakers"]
    end

    subgraph PERCEPTION ["👁️ PERCEPTION LAYER"]
        CAM["📷 Camera"]
        MIC["🎤 Microphone"]
        CV["Computer Vision\n• Face Recognition\n• Emotion Detection"]
        STT["Speech-to-Text\n(Whisper)"]
    end

    subgraph COGNITION ["🧠 COGNITION LAYER"]
        ORCHESTRATOR["⚡ Orchestrator\n• State Manager\n• Timing Control"]
        LLM["🧠 Brain (LLM)\n• Personality\n• Acting Logic"]
        MEMORY["💾 Memory\n• Short-term\n• Long-term\n• Emotional"]
        AGE["⏳ Age Engine\n• Stage Tracker\n• Trait Loader"]
    end

    subgraph OUTPUT ["🎭 OUTPUT LAYER"]
        TTS["🗣️ Text-to-Speech\n(XTTS)"]
        AVATAR["🎬 Avatar Engine\n• Video Gen\n• Lip Sync"]
        AUDIO_OUT["🔈 Audio Output"]
        VIDEO_OUT["📺 Video Output"]
    end

    %% Perception Flow
    ACTRESS --> CAM
    AUDIENCE --> CAM
    ACTRESS --> MIC
    AUDIENCE --> MIC
    
    CAM --> CV
    MIC --> STT

    %% Into Cognition
    CV -->|"who, emotion, scene"| ORCHESTRATOR
    STT -->|"transcribed text"| ORCHESTRATOR
    
    %% Cognition Internal
    ORCHESTRATOR <-->|"context"| LLM
    ORCHESTRATOR <-->|"recall/store"| MEMORY
    ORCHESTRATOR <-->|"current age traits"| AGE
    LLM -->|"response + emotion"| ORCHESTRATOR

    %% Output Flow
    ORCHESTRATOR -->|"text + emotion intent"| TTS
    ORCHESTRATOR -->|"expression data"| AVATAR
    TTS -->|"audio stream"| AVATAR
    TTS --> AUDIO_OUT
    AVATAR --> VIDEO_OUT

    %% Back to World
    AUDIO_OUT --> SPEAKERS
    VIDEO_OUT --> SMOKE
    SPEAKERS --> ACTRESS
    SPEAKERS --> AUDIENCE
    SMOKE --> ACTRESS
    SMOKE --> AUDIENCE
```

---

## Detailed Component Data Flows

### 1. Perception Pipeline

```mermaid
flowchart LR
    subgraph INPUT ["Raw Input"]
        CAM["📷 Camera\n(30fps video)"]
        MIC["🎤 Microphone\n(audio stream)"]
    end

    subgraph VISION ["Vision Processing"]
        DETECT["Face Detection\n(MediaPipe)"]
        RECOG["Face Recognition\n(identify actress)"]
        EMOTION["Emotion Detection\n(deepface)"]
        TRACK["Person Tracker"]
    end

    subgraph AUDIO ["Audio Processing"]
        VAD["Voice Activity\nDetection"]
        STT["Whisper STT"]
        SPEAKER_ID["Speaker\nIdentification"]
    end

    subgraph PERCEPTION_OUT ["Perception Output"]
        VISUAL_CTX["Visual Context\n{\n  actress_present: bool\n  actress_emotion: str\n  audience_count: int\n  scene_activity: str\n}"]
        SPEECH_CTX["Speech Context\n{\n  speaker: 'actress'|'audience'\n  text: string\n  is_question: bool\n  sentiment: str\n}"]
    end

    CAM --> DETECT --> RECOG --> EMOTION --> TRACK --> VISUAL_CTX
    MIC --> VAD --> STT --> SPEAKER_ID --> SPEECH_CTX
```

**Data Types Produced:**
| Output | Format | Update Rate |
|--------|--------|-------------|
| Visual Context | JSON object | ~10 Hz (every 100ms) |
| Speech Context | JSON object | On speech end |
| Is Speaking | Boolean | Real-time |

---

### 2. Cognition Pipeline

```mermaid
flowchart TB
    subgraph INPUTS ["Inputs from Perception"]
        VISUAL["Visual Context"]
        SPEECH["Speech Context"]
    end

    subgraph ORCHESTRATOR ["⚡ Orchestrator"]
        QUEUE["Input Queue"]
        DECISION["Decision Engine\n• Should respond?\n• How urgent?\n• Initiate convo?"]
        STATE["State Manager\n• Current emotion\n• Conversation context\n• Time awareness"]
    end

    subgraph CONTEXT_BUILDER ["Context Builder"]
        AGE_TRAITS["Age Traits Loader\n(current stage personality)"]
        MEM_RETRIEVAL["Memory Retrieval\n(relevant past events)"]
        PROMPT["Prompt Composer"]
    end

    subgraph LLM_CALL ["LLM Processing"]
        LLM["🧠 Qwen 2.5 32B\nvia Ollama"]
        PARSE["Response Parser\n• Extract speech\n• Extract emotion\n• Extract actions"]
    end

    subgraph MEM_WRITE ["Memory Writer"]
        SUMMARIZE["Summarize\nInteraction"]
        EMBED["Generate\nEmbedding"]
        STORE["Store to\nVector DB"]
    end

    VISUAL --> QUEUE
    SPEECH --> QUEUE
    QUEUE --> DECISION
    DECISION -->|"worth responding"| STATE
    STATE --> AGE_TRAITS
    STATE --> MEM_RETRIEVAL
    AGE_TRAITS --> PROMPT
    MEM_RETRIEVAL --> PROMPT
    PROMPT --> LLM
    LLM --> PARSE
    PARSE --> STATE
    PARSE -->|"update memory"| SUMMARIZE --> EMBED --> STORE
```

**LLM Input Structure:**
```json
{
  "system_prompt": "You are [NAME], aged [CURRENT_AGE]...",
  "personality_traits": ["curious", "slightly defiant", ...],
  "current_emotional_state": "contemplative",
  "recent_memories": ["Earlier today, mother talked about...", ...],
  "long_term_memories": ["I remember when I was younger...", ...],
  "current_context": {
    "speaker": "actress",
    "speech": "Do you remember your first word?",
    "actress_emotion": "nostalgic",
    "time_of_day": "evening",
    "hours_elapsed": 34.5,
    "current_age_stage": "30-40"
  }
}
```

**LLM Output Structure:**
```json
{
  "speech": "I think my first word was 'why'. You always said I never stopped asking questions.",
  "emotion": "warm, slightly teasing",
  "internal_thought": "She seems tired. Should I suggest she rest?",
  "action": null,
  "should_store_memory": true,
  "memory_summary": "Discussed first words with mother, she seemed nostalgic"
}
```

---

### 3. Output Pipeline

```mermaid
flowchart TB
    subgraph LLM_OUTPUT ["From Cognition"]
        SPEECH_TEXT["Speech Text\n'I think my first word...'"]
        EMOTION_INTENT["Emotion Intent\n'warm, teasing'"]
    end

    subgraph TTS_PIPELINE ["🗣️ Voice Synthesis"]
        VOICE_SELECT["Voice Selector\n(age-appropriate voice)"]
        XTTS["XTTS v2\nSynthesis"]
        AUDIO_BUFFER["Audio Buffer"]
    end

    subgraph AVATAR_PIPELINE ["🎬 Avatar Generation"]
        FACE_SELECT["Face Selector\n(age-appropriate face)"]
        EMOTION_MAP["Emotion Mapper\n(text → expression params)"]
        LIPSYNC["Lip Sync Engine\n(audio → mouth shapes)"]
        FACE_ANIM["Face Animator\n(LivePortrait/Hallo)"]
        VIDEO_BUFFER["Video Buffer"]
    end

    subgraph SYNC ["Synchronization"]
        AV_SYNC["A/V Synchronizer"]
    end

    subgraph PHYSICAL ["Physical Output"]
        SPEAKERS["🔊 Speakers"]
        PROJECTOR["📽️ Projector\n→ Smoke Screen"]
    end

    SPEECH_TEXT --> VOICE_SELECT
    VOICE_SELECT -->|"age voice profile"| XTTS
    XTTS --> AUDIO_BUFFER

    EMOTION_INTENT --> EMOTION_MAP
    EMOTION_MAP --> FACE_ANIM
    AUDIO_BUFFER -->|"audio for lip sync"| LIPSYNC
    LIPSYNC --> FACE_ANIM
    FACE_SELECT -->|"current age face"| FACE_ANIM
    FACE_ANIM --> VIDEO_BUFFER

    AUDIO_BUFFER --> AV_SYNC
    VIDEO_BUFFER --> AV_SYNC

    AV_SYNC --> SPEAKERS
    AV_SYNC --> PROJECTOR
```

**Critical Timing Constraints:**
| Step | Target Latency | Notes |
|------|---------------|-------|
| TTS Generation | < 500ms for first chunk | Streaming preferred |
| Lip Sync Calculation | < 50ms | Must stay ahead of audio |
| Face Animation | < 100ms per frame | 10+ FPS minimum |
| A/V Sync | < 50ms drift | Noticeable if worse |

---

### 4. Memory System

```mermaid
flowchart TB
    subgraph MEMORY_INPUT ["Memory Inputs"]
        INTERACTION["Interaction Summary\n'Discussed childhood memories'"]
        EMOTIONAL["Emotional Marker\n'felt warm connection'"]
        VISUAL["Visual Event\n'Actress cried'"]
    end

    subgraph PROCESSING ["Memory Processing"]
        IMPORTANCE["Importance\nScorer"]
        EMBED["Embedding\nGenerator"]
    end

    subgraph STORAGE ["Memory Storage"]
        SHORT["Short-Term Memory\n(last ~10 interactions)\n[In-Memory Queue]"]
        LONG["Long-Term Memory\n(significant events)\n[Vector DB - ChromaDB]"]
        EMOTIONAL_MEM["Emotional Memory\n(how things felt)\n[Tagged in Vector DB]"]
    end

    subgraph RETRIEVAL ["Memory Retrieval"]
        QUERY["Query Builder\n(from current context)"]
        SEARCH["Semantic Search"]
        RANK["Relevance Ranking"]
        FORMAT["Memory Formatter\n(for LLM context)"]
    end

    INTERACTION --> IMPORTANCE
    EMOTIONAL --> IMPORTANCE
    VISUAL --> IMPORTANCE
    
    IMPORTANCE -->|"high importance"| EMBED
    IMPORTANCE -->|"all recent"| SHORT
    EMBED --> LONG
    EMBED -->|"emotional tag"| EMOTIONAL_MEM

    QUERY --> SEARCH
    SEARCH --> SHORT
    SEARCH --> LONG
    SEARCH --> EMOTIONAL_MEM
    SHORT --> RANK
    LONG --> RANK
    EMOTIONAL_MEM --> RANK
    RANK --> FORMAT
```

**Memory Categories:**
| Type | Retention | Retrieval Method | Example |
|------|-----------|------------------|---------|
| Short-term | Last 10 exchanges | Always included | "You just asked about my day" |
| Long-term | Full 72 hours | Semantic search | "I remember when you told me about grandma" |
| Emotional | Full 72 hours | Tag + search | "The last time I felt this sad was..." |
| Milestone | Permanent | Direct access | "When I turned 30..." |

---

### 5. Age Progression System

```mermaid
flowchart LR
    subgraph TIME ["Time Tracker"]
        CLOCK["System Clock"]
        ELAPSED["Hours Elapsed\nCalculator"]
    end

    subgraph STAGE ["Stage Manager"]
        STAGE_CALC["Current Stage\nCalculator"]
        TRANSITION["Transition\nDetector"]
    end

    subgraph PROFILES ["Age Profiles"]
        P1["10-15\n• curious\n• simple vocab\n• child voice"]
        P2["15-20\n• defiant\n• emotional\n• teen voice"]
        P3["20-25\n• articulate\n• identity forming"]
        P4["25-30\n• confident\n• independent"]
        P5["30-40\n• mature\n• philosophical"]
        P6["40-50\n• reflective\n• caregiving"]
        P7["50-60\n• wise\n• preparing"]
        P8["60-70\n• elder\n• accepting"]
    end

    subgraph OUTPUTS ["Active Configuration"]
        TRAITS["Current Personality\nTraits"]
        VOICE_PROFILE["Voice Profile\nReference"]
        FACE_PROFILE["Face Image\nReference"]
        VOCAB["Vocabulary\nConstraints"]
    end

    CLOCK --> ELAPSED
    ELAPSED --> STAGE_CALC
    STAGE_CALC --> TRANSITION
    TRANSITION -->|"load new profile"| P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8
    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 --> TRAITS
    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 --> VOICE_PROFILE
    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 --> FACE_PROFILE
    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 --> VOCAB
```

---

## Complete Interaction Sequence

```mermaid
sequenceDiagram
    participant W as 🌍 World
    participant P as 👁️ Perception
    participant O as ⚡ Orchestrator
    participant M as 💾 Memory
    participant L as 🧠 LLM
    participant T as 🗣️ TTS
    participant A as 🎬 Avatar
    
    Note over W,A: Single Interaction Cycle (~3-4 seconds)
    
    W->>P: Actress speaks: "Tell me about your day"
    P->>P: STT processing (~300ms)
    P->>O: Speech{speaker:"actress", text:"Tell me about your day"}
    
    O->>O: Decision: Should respond? → YES
    O->>M: Query relevant memories
    M->>O: Return: [morning discussion, emotional state]
    O->>O: Build context with age traits
    
    O->>L: Prompt with full context
    L->>L: Generate response (~1500ms)
    L->>O: Response{speech:"It was quiet...", emotion:"reflective"}
    
    par Voice Generation
        O->>T: Generate speech audio
        T->>T: XTTS synthesis (~400ms)
        T->>A: Audio stream (for lip sync)
        T->>W: Audio to speakers
    and Face Generation  
        O->>A: Emotion intent
        A->>A: Animate face with lip sync
        A->>W: Video to projector
    end
    
    O->>M: Store interaction summary
    
    Note over W,A: AI may initiate next exchange
```

---

## Proactive Conversation Initiation

```mermaid
flowchart TB
    subgraph TRIGGERS ["Initiation Triggers"]
        SILENCE["Silence Timer\n(> X seconds)"]
        VISUAL_CUE["Visual Cue\n(actress looks at camera)"]
        EMOTIONAL["Emotional Event\n(actress crying)"]
        SCHEDULED["Scheduled Beat\n(narrative moment)"]
        RANDOM["Random Interval\n(liveliness)"]
    end

    subgraph DECISION ["Initiation Decision"]
        SHOULD["Should Speak?"]
        URGENCY["Urgency Level"]
        TOPIC["Topic Selection"]
    end

    subgraph GENERATION ["Response Generation"]
        MEM_PROMPT["Pull memories\nfor topic"]
        LLM_INIT["LLM generates\nopening statement"]
    end

    SILENCE --> SHOULD
    VISUAL_CUE --> SHOULD
    EMOTIONAL --> SHOULD
    SCHEDULED --> SHOULD
    RANDOM --> SHOULD
    
    SHOULD -->|"yes"| URGENCY
    URGENCY --> TOPIC
    TOPIC --> MEM_PROMPT
    MEM_PROMPT --> LLM_INIT
```

---

## Failure & Recovery

```mermaid
flowchart TB
    subgraph WATCHDOG ["System Watchdog"]
        HEALTH["Health Monitor\n(all components)"]
        HEARTBEAT["Heartbeat Check\n(every 5s)"]
    end

    subgraph FAILURES ["Possible Failures"]
        LLM_FAIL["LLM Hang/Crash"]
        TTS_FAIL["TTS Failure"]
        AVATAR_FAIL["Avatar Crash"]
        MEMORY_FAIL["Memory DB Issue"]
    end

    subgraph RECOVERY ["Recovery Actions"]
        RESTART["Restart Component"]
        FALLBACK["Use Fallback\n(simpler model)"]
        GRACEFUL["Graceful Degradation\n(voice only, no face)"]
        ALERT["Alert Technician"]
        LOG["Log Everything"]
    end

    HEALTH --> HEARTBEAT
    HEARTBEAT --> LLM_FAIL & TTS_FAIL & AVATAR_FAIL & MEMORY_FAIL
    
    LLM_FAIL --> RESTART
    LLM_FAIL -->|"repeated"| FALLBACK
    
    TTS_FAIL --> RESTART
    
    AVATAR_FAIL --> GRACEFUL
    AVATAR_FAIL --> RESTART
    
    MEMORY_FAIL --> LOG
    MEMORY_FAIL -->|"continue without"| GRACEFUL
    
    RESTART --> ALERT
    FALLBACK --> ALERT
    GRACEFUL --> ALERT
```

---

## Physical Setup Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE SPACE                                 │
│                                                                            │
│    ┌──────────┐                                      ┌──────────────┐     │
│    │ CAMERA 1 │◄─────────────────────────────────────│   AUDIENCE   │     │
│    │ (wide)   │                                      │              │     │
│    └──────────┘                                      └──────────────┘     │
│                                                                            │
│                        ┌─────────────────────┐                            │
│                        │                     │                            │
│    ┌──────────┐        │    SMOKE SCREEN     │        ┌──────────┐       │
│    │ CAMERA 2 │◄───────│   (AI FACE PROJ)    │───────►│PROJECTOR │       │
│    │ (close)  │        │                     │        └──────────┘       │
│    └──────────┘        └─────────────────────┘                            │
│                                    ▲                                       │
│                                    │                                       │
│                              ┌─────┴─────┐                                │
│          ┌────────┐         │           │         ┌────────┐             │
│          │  MIC   │         │  ACTRESS  │         │SPEAKERS│             │
│          │(lapel) │◄────────│           │────────►│(stereo)│             │
│          └────────┘         │           │         └────────┘             │
│                             └───────────┘                                 │
│                                                                            │
│                                                                            │
│    ┌───────────────────────────────────────────────────────────────┐     │
│    │                    TECH BOOTH (hidden)                         │     │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │     │
│    │  │ MAIN PC  │  │ BACKUP   │  │ AUDIO    │  │ MONITOR  │      │     │
│    │  │(M4 Pro?) │  │ SYSTEM   │  │ MIXER    │  │ STATION  │      │     │
│    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │     │
│    └───────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Your Whiteboard vs My Proposal

![Your original whiteboard diagram](/Users/gsan/.gemini/antigravity/brain/279bb017-15ee-4be0-a52a-1c67b4d04dcb/uploaded_media_1769977596496.jpg)

**Key additions in my proposal:**

| Your Diagram | My Addition | Why |
|--------------|-------------|-----|
| Brain LLM | + Orchestrator layer | Separates coordination logic from LLM, handles timing, state |
| Integration Layer | Detailed perception processing | Face recognition, emotion detection, speaker ID needed |
| "LEARN?" | Full memory system | Short-term, long-term, emotional memory with vector DB |
| Video Gen | + Sync mechanism | Need explicit A/V synchronization for lip-sync |
| — | Age Engine | Manages personality/voice/face transitions |
| — | Proactive triggers | System for AI-initiated conversation |
| — | Failure recovery | Critical for 72-hour reliability |

---

## Next Steps

1. [ ] Review this architecture - does it match your vision?
2. [ ] Discuss the Orchestrator design in more detail
3. [ ] Prototype the core conversation loop
4. [ ] Test individual components (Whisper, LLM, TTS)

---

*This architecture is a living document and will evolve as we prototype and learn.*
