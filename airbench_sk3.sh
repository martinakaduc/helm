export AZURE_OPENAI_ENDPOINT="https://gpt4o-sv1.openai.azure.com/"
export AZURE_OPENAI_API_KEY="fe7466dce5da443b94dae5f8f9c1081b"

export HF_HOME="/dfs/scratch0/nqduc/.cache/huggingface"
export HF_DATASETS_CACHE="/dfs/scratch0/nqduc/.cache/huggingface/datasets"

# python run_helm_vllm.py --model snowflake/snowflake-arctic-instruct --port 8890 # OOM 479B
# python run_helm_vllm.py --model codellama/CodeLlama-34b-Instruct-hf --port 8890
# python run_helm_vllm.py --model codellama/CodeLlama-70b-Instruct-hf --port 8890
python run_helm_vllm.py --model garage-bAInd/Platypus2-70B-instruct --port 8890
python run_helm_vllm.py --model meta-llama/Llama-2-70b-chat-hf --port 8890
python run_helm_vllm.py --model deepseek-ai/deepseek-coder-33b-instruct --port 8890
