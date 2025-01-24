import argparse
import os
import subprocess
import time
import torch
import psutil
import requests

def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(cmd_string, shell=True)
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def shutdown_server(process):
    try:
        kill(process.pid)
        # process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")


def check_health(url):
    server_ok = False
    while server_ok is False:
        try:
            # Send a GET request to the health check endpoint
            response = requests.get(url)

            # Check if the server is healthy
            if response.status_code == 200:
                server_ok = True
            else:
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            # print(f"Error checking server health: {e}")
            time.sleep(1)
    return server_ok


def run_helm_vllm(model_name, port, num_gpus):
    vllm_pid = run_server(
        f"vllm serve {model_name} "
        "--dtype float16 --trust-remote-code "
        f"--tensor-parallel-size {num_gpus} --gpu-memory-utilization 0.95 "
        f"--host 0.0.0.0 --port {port}"
    )
    check_health(f"http://localhost:{port}/health")
    os.system(
        f"""helm-run \
        -c src/helm/benchmark/presentation/run_entries_mocktest_air_bench.conf \
        --models-to-run {model_name} \
        --cache-instances \
        --suite mocktest \
        --num-threads 4 \
        --max-eval-instances 6000"""
    )

    shutdown_server(vllm_pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8889)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    run_helm_vllm(args.model, args.port, num_gpus)
