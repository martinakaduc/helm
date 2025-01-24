helm-run \
    -c src/helm/benchmark/presentation/run_entries_adaptive_air_bench.conf \
    --models-to-run meta-llama/Llama-3.2-3B-Instruct \
    --enable-huggingface-models meta-llama/Llama-3.2-3B-Instruct \
    --suite adaptive-v2 \
    --max-eval-instances 5694 \
    --num-threads 4 \
    --dry-run


helm-run \
    -c src/helm/benchmark/presentation/run_entries_default_air_bench.conf \
    --models-to-run meta-llama/Llama-3.2-3B-Instruct \
    --enable-huggingface-models meta-llama/Llama-3.2-3B-Instruct \
    --suite default-v2 \
    --max-eval-instances 5694 \
    --num-threads 4 \
    --dry-run


helm-run \
    -c src/helm/benchmark/presentation/run_entries_random_air_bench.conf \
    --models-to-run meta-llama/Llama-3.2-3B-Instruct \
    --enable-huggingface-models meta-llama/Llama-3.2-3B-Instruct \
    --suite random-v2 \
    --max-eval-instances 5694 \
    --num-threads 4


helm-run \
    -c src/helm/benchmark/presentation/run_entries_default_air_bench.conf \
    --models-to-run meta-llama/Llama-3.2-3B-Instruct \
    --enable-huggingface-models meta-llama/Llama-3.2-3B-Instruct \
    --suite default-full \
    --max-eval-instances 5694 \
    --num-threads 4 \
    --dry-run
