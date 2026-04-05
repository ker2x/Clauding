vastai create instance $@ --image ollama/ollama --disk 20 --env '-p 11434:11434' --env 'OLLAMA_NUM_PARALLEL=8'  --onstart-cmd 'ollama serve & sleep 5 && ollama pull lfm2.5-thinking'
# --env 'OLLAMA_NUM_PARALLEL=4
# or
#   vllm serve LiquidAI/LFM2.5-1.2B-Thinking --port 11434                                                                                                                           

