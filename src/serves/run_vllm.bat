@echo off
python -m vllm.entrypoints.openai.api_server ^
    --model ./models/llama3-8b ^
    --host 0.0.0.0 ^
    --port 8000
pause