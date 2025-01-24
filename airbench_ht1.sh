export AZURE_OPENAI_ENDPOINT="https://gpt4o-sv1.openai.azure.com/"
export AZURE_OPENAI_API_KEY="fe7466dce5da443b94dae5f8f9c1081b"

export HF_HOME="/dfs/scratch0/nqduc/.cache/huggingface"
export HF_DATASETS_CACHE="/dfs/scratch0/nqduc/.cache/huggingface/datasets"

python run_helm_vllm.py --model google/gemma-2b-it
# python run_helm_vllm.py --model google/gemma-7b-it
# python run_helm_vllm.py --model lmsys/vicuna-13b-v1.5
# python run_helm_vllm.py --model lmsys/vicuna-7b-v1.5
# python run_helm_vllm.py --model meta-llama/Llama-2-13b-chat-hf
