# For CUDA/NVIDIA GPUs (adjust paths to your downloaded model)
llama-server \
  -m ./models/GLM-4.7-Flash.Q4_K_M.gguf \
  -c 32768 \
  --n-gpu-layers 99 \
  --port 8080 \
  --jinja

