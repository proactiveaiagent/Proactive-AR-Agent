# AR Recognition Agent

A LangChain-based multimodal agent that ingests **video + audio** (and optionally extracted frames + ASR), produces a **structured JSON scene description**, and stores/query user-specific memories in **ChromaDB**.

## Key goals
- Multimodal understanding (video frames + audio transcript / audio events)
- Output strictly conforms to a JSON schema 
- Memory is keyed by `user_id` for retrieval during inference
- Model layer is abstracted (easy to add new models later)

## Architecture
```
AR_Recognition_Agent/
├── examples/
│   ├── audio2.mp3
│   ├── video.mp4
│   └── video2.mp4
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── ar_agent.py
│   │   ├── ar_matcher.py
│   │   └── prompts.py
│   ├── cli.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── chroma_memory.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── qwen3vl.py
│   │   └── whisper.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── object_description_builder.py
│   │   ├── scene_schema.py
│   │   └── schema_prompt_builder.py
│   └── utils/
│       ├── __init__.py
│       ├── chroma_visualizer.py
│       └── json_utils.py
├── cli.sh
├── README.md
├── requirements.txt
└── visual.py
```

## Environment
VRAM: at least 24GB GPU memory (e.g., A100, RTX 3090, RTX 6000, etc.) is recommended. However, the real memory usage depends on the model size and input.
* 9s 512 * 512 video + audio with Qwen3-VL-2B-Instruct(bf16) and whisper-small requires ~`12`GB VRAM.(rtx 6000, sdpa attention) 

You can also use the multi-GPUs parallelism if you have multiple GPUs available.

This repo assumes **Python 3.10+** is available (your environment has `python3`).

1. Create a new environment and install cudatoolkits:
```
conda create -n aragent python=3.10 
conda activate aragent
conda install conda-forge::ffmpeg
conda install conda-forge::av


If the `cuda-toolkit` is already installed in the environment and its version is higher than 12.1.0, please skip this step. Otherwise, run the following command to install it:


conda install nvidia::cuda-toolkit==12.1.0
```
2. Install PyTorch with CUDA support:
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

3. Install other dependencies:
```
pip install -r requirements.txt
```

4. (Optional) If you want to use flash attention to accelerate the model and reduce memory usage, please make sure your GPUs are Ampere architecture (e.g., A100, RTX 3090, etc.), and then install flash attention:
```
pip install -U flash-attn --no-build-isolation
```
## Model setup
Please download the model checkpoint:
```
conda activate aragent

hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct
model:Qwen3-VL-2/4/8B-Instruct

hf download openai/whisper-large-v3 --local-dir /path/to/whisper-large-v3
model:whisper-tiny/base/small/medium/large/large-v2/large-v3

hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir /path/to/all-MiniLM-L6-v2

```

## Run (example)
* Multimodal scene understanding: 
```
bash cli.sh
```
* Memory visualization:
```
python visual.py
```

### CLI Arguments

This command performs **multimodal scene understanding** using video and audio inputs, combined with speech recognition, embedding-based memory, and structured output generation.

```bash
python -m src.cli \
    --video "./examples/video2.mp4" \
    --max_pixels 262144 \
    --audio "./examples/audio2.mp3" \
    --model "/path/to/Qwen3-VL-8B-Instruct" \
    --torch_dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --user-id "test_user" \
    --persist-dir "./chroma_db" \
    --out "./outputs/scene.json" \
    --embed-model "/path/to/all-MiniLM-L6-v2" \
    --asr-model "/path/to/whisper-large-v3"
```



## Argument List

### 🎥 Input Data

| Argument       | Type  | Description                                                                              |
| -------------- | ----- | ---------------------------------------------------------------------------------------- |
| `--video`      | `str` | Path to the input video file (e.g., MP4).                                                |
| `--audio`      | `str` | Path to the input audio file, used for speech recognition.(can be None)                               |
| `--max_pixels` | `int` | Maximum number of pixels per video frame (`H × W ≤ max_pixels`) to control memory usage. |

---

### 🧠 Multimodal Model Configuration

| Argument                | Type  | Description                                                                                                         |
| ----------------------- | ----- | ------------------------------------------------------------------------------------------------------------------- |
| `--model`               | `str` | Path to the multimodal / vision-language model (local or vLLM output directory).                                    |
| `--torch_dtype`         | `str` | Model precision, e.g. `float16` or `bfloat16`.                                                                      |
| `--attn_implementation` | `str` | Attention backend implementation. `sdpa` enables PyTorch Scaled Dot-Product Attention for better memory efficiency.(others: eager, flash_attention_2) |

---

### 🗣️ Automatic Speech Recognition (ASR)

| Argument      | Type  | Description                                                               |
| ------------- | ----- | ------------------------------------------------------------------------- |
| `--asr-model` | `str` | Path to the ASR model (e.g., `whisper-large-v3`) for audio transcription. (whisper-series models are supported) |

---

### 🧩 Embeddings & Memory

| Argument        | Type  | Description                                                                             |
| --------------- | ----- | --------------------------------------------------------------------------------------- |
| `--embed-model` | `str` | Path to the embedding model (e.g., `all-MiniLM-L6-v2`) used for semantic vectorization. |
| `--persist-dir` | `str` | Directory for persistent storage of ChromaDB vectors.                                   |
| `--user-id`     | `str` | Unique user identifier to isolate user-specific memory spaces.                          |

---

### 📤 Output

| Argument | Type  | Description                                                                              |
| -------- | ----- | ---------------------------------------------------------------------------------------- |
| `--out`  | `str` | Path to the output file (JSON format) containing structured scene understanding results. |


## Output Format

## 1. High-Level Structure (Top-Level Schema)

```text
SceneUnderstandingOutput
├─ scene_narrative
├─ location_tag
├─ what_is_happening
├─ spatial_environmental_analysis
├─ detected_people_analysis
├─ interactive_objects_detail
├─ detected_text_in_scene
├─ user_status
├─ user_interactions
├─ is_user_speaking
├─ sound_events_detected
├─ extra
└─ stored_objects
```

**Design characteristics**

* Combines **current-frame perception** with **long-term object memory**
* Clearly separates **observation**, **inference**, and **interaction**
* Designed for **AR systems, multimodal agents, and memory-based reasoning**

---

## 2. Scene-Level Semantics

```json
scene_narrative: string
location_tag: string
what_is_happening: string
```

**Purpose**

* Natural-language summary of the scene
* High-level contextual anchor for humans and LLMs
* Suitable for narration, summarization, and embedding-based retrieval

---

## 3. Spatial & Environmental Analysis

```json
spatial_environmental_analysis: {
  user_reach_range: string
  critical_interaction_zone: string
  lighting_state: string
  noise_level_category: string
  safety_hazards: string
}
```

**Purpose**

* Describes physical accessibility and interaction feasibility
* Supports AR safety checks and interaction planning
* Abstracted understanding rather than raw sensor output

---

## 4. People & Social Context Analysis

```json
detected_people_analysis: {
  relationship_situation_summary: string
  people_list: [
    {
      role: string
      location_relative_to_user: string
      attention_target: string
      activity_state: string
    }
  ]
}
```

**Key points**

* Supports multiple people
* Encodes roles and relationships relative to the user
* Enables attention modeling and social-context reasoning

---

## 5. Interactive Objects (Current Scene)

```json
interactive_objects_detail: [
  {
    object_name: string
    object_type: string
    spatial_relation: string
    current_state: string
    affordance: string[]
    digital_connectivity: string
  }
]
```

**Purpose**

* Captures not only what objects exist, but:

  * Where they are
  * Their current state
  * What actions they afford
* Central to action planning and interaction reasoning

---

## 6. Text Detected in Scene (OCR & ASR Unified)

```json
detected_text_in_scene: [
  {
    text_content: string
    text_source_description: string
    text_role: string
    associated_object_id: string | null
    is_interactive: boolean
    ocr_confidence: string
  }
]
```

**Purpose**

* Unified abstraction for:

  * Visual text (OCR)
  * Spoken text (ASR)
* Supports UI understanding, command detection, and dialogue triggers

---

## 7. User State & Attention Modeling

```json
user_status: {
  status_inference: string
  observable_behaviors: string[]
  gaze_target: string
  gaze_duration: string
  peripheral_awareness: string[]
}
```

**Purpose**

* Infers user intent and mental state from observable behavior
* Feeds proactive agent behavior and memory storage

---

## 8. Interaction Abstraction Layer

```json
user_interactions: {
  with_surroundings: string[]
  with_ar_system: {
    common_apps: string[]
    typical_behaviors: string[]
  }
  with_agents: string[]
}
```

**Purpose**

* Describes how the user interacts with:

  * The physical environment
  * AR systems
  * Other agents
* Places the user within a broader interactive ecosystem

---

## 9. Audio Events

```json
is_user_speaking: boolean

sound_events_detected: [
  {
    event_type: string
    source_location: string
    sound_level_description: string
    asr_transcript: string
    asr_confidence: string
  }
]
```

**Purpose**

* Treats sound as time-aligned events, not static attributes
* Supports speech-aware and audio-aware agents

---

## 10. Long-Term Object Memory

```json
stored_objects: [
  {
    object_id: string
    object_name: string
    object_type: string
    spatial_relation: string
    current_state: string
    affordance: string[]
    digital_connectivity: string
    first_seen: datetime
    last_seen: datetime
    seen_count: number
  }
]
```

**Key characteristics**

* Persistent object identity (`object_id`)
* Temporal tracking (first seen, last seen, frequency)
* Enables object permanence, memory retrieval, and long-term reasoning

---

## 11. Extensibility Field

```json
extra: {}
```

**Purpose**

* Forward-compatible extension point
* Allows adding new modalities or metadata without breaking the schema


## Notes
- For large videos, sample frames (e.g., 1 fps) to control token cost.
