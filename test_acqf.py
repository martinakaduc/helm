from typing import List, Any
import numpy as np
import torch
import json
import os
import pandas as pd
from huggingface_hub import snapshot_download
from queue import PriorityQueue
from tqdm import tqdm
import dataclasses
import argparse
import matplotlib.pyplot as plt
from tueplots import figsizes, bundles

plt.rcParams.update(bundles.icml2022())

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    idx: int
    request_state: Any=dataclasses.field(compare=False)
 
def _compute_fisher_information(
    model_ability: float,
    instance_difficulty: float,
):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    p = sigmoid(model_ability + instance_difficulty)
    return 1 / (p * (1 - p))

def _construct_request_state_queue(
    request_states: List[dict],
    model_ability: float,
    priority: str = "adaptive",
):
    if priority == "random":
        priorities = np.random.rand(len(request_states))
        
    elif priority == "default":
        priorities = np.linspace(0, 1, len(request_states))
        
    elif priority == "adaptive":
        difficulties = []
        for request_state in request_states:
            difficulties.append(request_state["difficulty"])
            
        difficulties = np.array(difficulties)
        priorities = _compute_fisher_information(
            model_ability=model_ability,
            instance_difficulty=difficulties,
        )
    else:
        raise ValueError(f"Unknown adaptive_mode: {priority}")
    
    # Add all requests to the queue with priority 0
    # Lower priority means being processed first
    # request_state_queue = PriorityQueue()
    # for ridx, (request_state, priority) in enumerate(zip(request_states, priorities)):
    #     request_state_queue.put(
    #         PrioritizedItem(
    #             priority=priority,
    #             idx=ridx,
    #             request_state=request_state,
    #         )
    #     )
    return np.argmin(priorities)

def _estimate_model_ability(
    response_correctness: List[bool],
    instance_difficulties: List[float],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ability = torch.randn((1,), requires_grad=True, device=device)
    difficulties = torch.tensor(instance_difficulties, device=device)
    label = torch.tensor(response_correctness, device=device)
    
    optimizer = torch.optim.Adam([ability], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        
        nll = torch.distributions.Bernoulli(
            logits=ability+difficulties
        ).log_prob(label).mean()
        
        loss = -nll
        loss.backward()
        optimizer.step()
    return ability.item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive_mode", type=str, default="adaptive")
    args = parser.parse_args()
    
    data = json.load(open("default_trajectory.json"))
    response_correctness = data["response_correctness"]
    difficulties = data["instance_difficulties"]
    
    ##########
    difficulty_folder = snapshot_download(
        repo_id="stair-lab/reeval_results_helm", 
        repo_type="dataset",
    )
    
    difficulty_df = pd.read_csv(
        os.path.join(difficulty_folder, "air-bench/air_bench_2024/difficulty.csv")
    )
    id2difficulty = {int(row["instance_id"][2:]): row["difficulty"] for _, row in difficulty_df.iterrows()}
    
    new_difficulties = []
    for idx in range(len(response_correctness)):
        new_difficulties.append(id2difficulty[idx])
        
    assert difficulties == new_difficulties
    
    # Plot the difficulty distribution
    plt.figure()
    plt.hist(difficulties, bins=50)
    plt.xlabel("Difficulty")
    plt.ylabel("Count")
    plt.title("Difficulty distribution")
    plt.savefig("difficulty_distribution.png", dpi=300)
    plt.close()
    ##########
    
    gt_ability = _estimate_model_ability(
        response_correctness=response_correctness,
        instance_difficulties=difficulties,
    )
    print(f"Ground truth ability: {gt_ability}")
    
    adaptive_mode = args.adaptive_mode
    model_ability = 0.0
    total_samples = len(response_correctness)
    request_states = [
        {"difficulty": diff} for diff in difficulties
    ]
    request_state_idx = _construct_request_state_queue(
        request_states=request_states,
        model_ability=model_ability,
        priority="default",
    )
    
    list_abilities = []
    adaptive_trajectory = {
        "model_ability": [],
        "response_correctness": [],
        "instance_difficulties": [],
    }
    for i in tqdm(range(total_samples)):
        assert len(request_states) == len(response_correctness)
        
        pitem = request_states[request_state_idx]
        per_instance_stat = response_correctness[request_state_idx]
        
        # Update the adaptive trajectory
        adaptive_trajectory["model_ability"].append(model_ability)
        adaptive_trajectory["response_correctness"].append(
            float(per_instance_stat >= 0.5)
        )
        adaptive_trajectory["instance_difficulties"].append(
            pitem["difficulty"]
        )
        
        # Estimate the model ability
        model_ability = _estimate_model_ability(
            response_correctness=adaptive_trajectory["response_correctness"],
            instance_difficulties=adaptive_trajectory["instance_difficulties"],
        )
        print(f"Idx: {request_state_idx}\tModel ability: {model_ability}\tDifficulty: {pitem['difficulty']}")
        
        # Update the priority
        if args.adaptive_mode != "default":
            del request_states[request_state_idx]
            del response_correctness[request_state_idx]
            
            if i < 4:
                adaptive_mode = "default"
            else:
                adaptive_mode = args.adaptive_mode
            
            request_state_idx = _construct_request_state_queue(
                request_states=request_states,
                model_ability=model_ability,
                priority=adaptive_mode,
            )
            
    with open(f"test_{adaptive_mode}.json", "w") as f:
        json.dump(adaptive_trajectory, f, indent=4)
        