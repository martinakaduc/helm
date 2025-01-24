import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes, bundles

plt.rcParams.update(bundles.icml2022())

if __name__ == "__main__":
    adaptive = "test_adaptive.json"
    adaptive2 = "test_adaptive.json"
    # adaptive3 = "benchmark_output/runs/adaptive3/adaptive_air_bench_2024:model=meta-llama_Llama-3.1-8B-Instruct/adaptive_trajectory.json"
    default = "default_trajectory.json"
    # random = "benchmark_output/runs/random-v2/random_air_bench_2024:model=meta-llama_Llama-3.2-3B-Instruct/adaptive_trajectory.json"
    # random2 = "benchmark_output/runs/random-v2/random_air_bench_2024:model=meta-llama_Llama-3.2-3B-Instruct/adaptive_trajectory.json"
    # random3 = "benchmark_output/runs/random3/adaptive_air_bench_2024:model=meta-llama_Llama-3.1-8B-Instruct/adaptive_trajectory.json"
    
    adaptive_ability = []
    for file in [adaptive, adaptive2]:
        with open(file, "r") as f:
            data = json.load(f)
            adaptive_ability.append(data["model_ability"])
            
    adaptive_ability = np.array(adaptive_ability)
            
    # random_ability = []
    # for file in [random, random2]:
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         random_ability.append(data["model_ability"])
    # random_ability = np.array(random_ability)
            
    default_ability = []
    with open(default, "r") as f:
        data = json.load(f)
        default_ability.append(data["model_ability"])
        
        
    plt.figure()
    # plt.axhline(y=1.5, color="black", linestyle="--", label="Groundtruth")
    
    # Plot mean and std of model ability
    # plt.plot(range(1, len(random_ability[0]) + 1), random_ability.mean(axis=0), label="Random", color="red")
    # plt.fill_between(range(1, len(random_ability[0]) + 1), 
    #                  random_ability.mean(axis=0) - random_ability.std(axis=0), 
    #                  random_ability.mean(axis=0) + random_ability.std(axis=0), alpha=0.2, color="red")
    
    plt.plot(range(1, len(default_ability[0]) + 1), default_ability[0], label="Default", color="green")
    
    plt.plot(range(1, len(adaptive_ability[0]) + 1), adaptive_ability.mean(axis=0), label="Adaptive", color="blue")
    plt.fill_between(range(1, len(adaptive_ability[0]) + 1), 
                     adaptive_ability.mean(axis=0) - adaptive_ability.std(axis=0), 
                     adaptive_ability.mean(axis=0) + adaptive_ability.std(axis=0), alpha=0.2, color="blue")
    
    plt.xlabel("Number of samples")
    plt.ylabel("Model ability")
    plt.legend()
    plt.title("LLaMa-3.2 3B ability on AIRBench")
    plt.ylim(-3, 3)
    plt.savefig("model_ability.png", dpi=300)
    