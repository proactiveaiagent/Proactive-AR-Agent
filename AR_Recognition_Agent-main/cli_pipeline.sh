python -m src.cli \
    --video "./examples/clip1.mp4" \
    --max_pixels 262144 \
    --audio "./examples/clip_audio1.mp3" \
    --model "/userhome/cs/u3651279/data/vllm/Qwen3-VL-8B-Instruct" \
    --torch_dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --user-id "test_user_2" \
    --persist-dir "./chroma_db" \
    --out "./outputs/scene1.json" \
    --embed-model "/userhome/cs/u3651279/data/emb/all-MiniLM-L6-v2" \
    --asr-model "/userhome/cs/u3651279/data/emb/whisper-large-v3"

python -m src.cli \
    --video "./examples/clip2.mp4" \
    --max_pixels 262144 \
    --audio "./examples/clip_audio2.mp3" \
    --model "/userhome/cs/u3651279/data/vllm/Qwen3-VL-8B-Instruct" \
    --torch_dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --user-id "test_user_2" \
    --persist-dir "./chroma_db" \
    --out "./outputs/scene2.json" \
    --embed-model "/userhome/cs/u3651279/data/emb/all-MiniLM-L6-v2" \
    --asr-model "/userhome/cs/u3651279/data/emb/whisper-large-v3"

python -m src.cli \
    --video "./examples/clip3.mp4" \
    --max_pixels 262144 \
    --audio "./examples/clip_audio3.mp3" \
    --model "/userhome/cs/u3651279/data/vllm/Qwen3-VL-8B-Instruct" \
    --torch_dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --user-id "test_user_2" \
    --persist-dir "./chroma_db" \
    --out "./outputs/scene3.json" \
    --embed-model "/userhome/cs/u3651279/data/emb/all-MiniLM-L6-v2" \
    --asr-model "/userhome/cs/u3651279/data/emb/whisper-large-v3"