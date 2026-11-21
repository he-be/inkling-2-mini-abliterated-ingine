@echo off
rem ingine OpenAI server on the RTX 5080. serve.bat <max_ctx> <host_experts> [port]
set CTX=%1
set HOSTX=%~2
set PORT=%3
if "%PORT%"=="" set PORT=8003
call C:\LLM\ingine-win\run.bat -m ingine.server --engine --model-dir C:\LLM\models\Inkling-2-Mini-hf ^
  --ablation C:\LLM\models\Inkling-2-Mini-hf\inkling-2-mini-abliterated.safetensors --model-id inkling-2-mini-abliterated ^
  --max-ctx %CTX% --dense-fp8 --kv-fp8 --host-experts %HOSTX% --host-embed ^
  --prefill-chunk 1024 --prefill-group 16 --full-segment 16384 --host 127.0.0.1 --port %PORT%
