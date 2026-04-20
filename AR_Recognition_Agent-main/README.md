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
│   ├── clip_audio1.mp3
│   ├── clip_audio2.mp3
│   ├── clip_audio3.mp3
│   ├── clip1.mp4
│   ├── clip2.mp4
│   ├── clip3.mp4
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

## Output Format Specification

This document defines the structured JSON output produced by the system for **scene understanding, object perception, people analysis, audio event recognition, and long-term memory matching**.

The format is strictly aligned with the system’s internal **Pydantic data models** and is designed to support:

* Multimodal reasoning
* AR / embodied AI agents
* Memory persistence and identity matching
* Downstream LLM or planning modules

---

## Top-Level Structure

```json
{
  "scene": { ... },
  "objects": { ... },
  "people": { ... },
  "extra": { ... },

  "stored_objects": [ ... ],
  "matched_people": [ ... ]
}
```

---

## 1. `scene` — Global Scene Description

Provides a **holistic understanding of the environment**, including context, activities, spatial constraints, and safety factors.

### Schema

```json
"scene": {
  "thought": string,
  "scene_narrative": string,
  "location_tag": string,
  "what_is_happening": string,
  "spatial_environmental_analysis": { ... },
  "detected_text_in_scene": [ ... ]
}
```

### Field Descriptions

| Field                    | Type   | Description                                                                                                      |
| ------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------- |
| `thought`                | string | Internal reasoning or abstract interpretation of the scene. Intended for agent cognition rather than UI display. |
| `scene_narrative`        | string | Natural-language summary describing the scene as a whole.                                                        |
| `location_tag`           | string | Semantic location label (e.g., `home_desk`, `office`, `outdoor_street`).                                         |
| `what_is_happening`      | string | Concise description of the primary ongoing activity.                                                             |
| `detected_text_in_scene` | array  | OCR-detected text elements in the environment.                                                                   |

---

### 1.1 `spatial_environmental_analysis`

Describes **interaction constraints and environmental context** relevant to safety and planning.

```json
"spatial_environmental_analysis": {
  "user_reach_range": string,
  "critical_interaction_zone": string,
  "lighting_state": string,
  "noise_level_category": string,
  "safety_hazards": string
}
```

| Field                       | Description                            |
| --------------------------- | -------------------------------------- |
| `user_reach_range`          | Physical area reachable by the user    |
| `critical_interaction_zone` | Area most relevant for interaction     |
| `lighting_state`            | Lighting condition (e.g., bright, dim) |
| `noise_level_category`      | Ambient noise level                    |
| `safety_hazards`            | Potential risks in the environment     |

---

### 1.2 `detected_text_in_scene`

Represents text detected via OCR.

```json
{
  "text_content": string,
  "text_source_description": string,
  "text_role": string,
  "associated_object_id": string | null,
  "is_interactive": boolean,
  "ocr_confidence": string
}
```

| Field                     | Description                                   |
| ------------------------- | --------------------------------------------- |
| `text_content`            | Detected text content                         |
| `text_source_description` | Where the text appears                        |
| `text_role`               | Functional role (label, warning, instruction) |
| `associated_object_id`    | Linked object if applicable                   |
| `is_interactive`          | Whether the text implies interaction          |
| `ocr_confidence`          | OCR confidence level                          |

---

## 2. `objects` — Interactive Object Description

Describes **objects relevant to user interaction**.

```json
"objects": {
  "interactive_objects_detail": [ ... ]
}
```

---

### 2.1 `interactive_objects_detail`

```json
{
  "thought": string,
  "object_name": string,
  "object_type": string,
  "spatial_relation": string,
  "current_state": string,
  "affordance": [ string ],
  "digital_connectivity": string
}
```

| Field                  | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `thought`              | Internal reasoning explaining object relevance |
| `object_name`          | Human-readable object name                     |
| `object_type`          | Object category                                |
| `spatial_relation`     | Position relative to the user                  |
| `current_state`        | Observable state                               |
| `affordance`           | Possible actions enabled by the object         |
| `digital_connectivity` | Network or device connectivity status          |

---

## 3. `people` — People and Social Understanding

Analyzes **people in the scene**, their roles, activities, and interaction with the user.

```json
"people": {
  "detected_people_analysis": { ... },
  "user_status": { ... },
  "user_interactions": { ... },
  "is_user_speaking": boolean | null,
  "sound_events_detected": [ ... ]
}
```

---

### 3.1 `detected_people_analysis`

```json
{
  "relationship_situation_summary": string,
  "people_list": [ ... ]
}
```

#### `people_list` Item

```json
{
  "thought": string,
  "people_id": string,
  "role": string,
  "location_relative_to_user": string,
  "attention_target": string,
  "activity_state": string,
  "kinship_term": string
}
```

| Field              | Description                                  |
| ------------------ | -------------------------------------------- |
| `people_id`        | Unique person identifier                     |
| `role`             | Semantic role (User, Friend, Stranger, etc.) |
| `attention_target` | Object or person being focused on            |
| `activity_state`   | Current action                               |
| `kinship_term`     | Social relationship descriptor               |

---

### 3.2 `user_status`

Captures **inferred mental, physical, and attentional state** of the user.

```json
{
  "status_inference": string,
  "observable_behaviors": [ string ],
  "gaze_target": string,
  "gaze_duration": string,
  "peripheral_awareness": [ string ]
}
```

---

### 3.3 `user_interactions`

```json
{
  "with_surroundings": [ string ],
  "with_ar_system": {
    "common_apps": [ string ],
    "typical_behaviors": [ string ]
  },
  "with_agents": [ string ]
}
```

---

### 3.4 `sound_events_detected`

Describes **non-verbal and verbal audio events**.

```json
{
  "event_type": string,
  "source_location": string,
  "sound_level_description": string,
  "asr_transcript": string | null,
  "asr_confidence": string | null
}
```

---

## 4. `stored_objects` — Long-Term Object Memory

Persistent memory entries for objects tracked across time.

```json
{
  "object_id": string,
  "object_name": string,
  "object_type": string,
  "spatial_relation": string,
  "current_state": string,
  "affordance": [ string ],
  "digital_connectivity": string,
  "first_seen": string,
  "last_seen": string,
  "seen_count": number
}
```

Used for:

* Object re-identification
* Temporal reasoning
* Usage statistics

---

## 5. `matched_people` — Identity Matching Results

Results of person identity matching against stored memory.

```json
{
  "person_id": string,
  "is_match": boolean,
  "confidence": number,
  "reasoning": string,
  "stored_person": { ... } | null
}
```

---

### 5.1 `stored_person`

```json
{
  "person_id": string,
  "name": string | null,
  "role": string,
  "kinship_term": string,
  "relationship_notes": string,
  "first_seen": string,
  "last_seen": string,
  "seen_count": number,
  "interaction_history": [ string ],
  "typical_locations": [ string ]
}
```

---

## 6. `extra` — Extension Field

```json
"extra": { }
```

Reserved for:

* Model-specific metadata
* Experimental outputs
* Future extensions



## Notes
- For large videos, sample frames (e.g., 1 fps) to control token cost.
