vastai create instance $@ --image vllm/vllm-openai:latest --disk 30 --env '-p 8000:8000' --onstart-cmd 'vllm serve LiquidAI/LFM2.5-1.2B-Thinking --port 8000 --served-model-name lfm2.5-thinking --max-num-seqs 32'
# --served-model-name matches config.py TEACHER_MODEL so oracle works unchanged
# --max-num-seqs 32 limits concurrent sequences (tune based on GPU VRAM)
# Add --env 'HUGGING_FACE_HUB_TOKEN=hf_...' if the model requires auth
# vLLM endpoint: http://<ip>:8000  (port differs from Ollama's 11434)
# vLLM handles concurrency natively — no need to duplicate URLs in TEACHER_URLS
