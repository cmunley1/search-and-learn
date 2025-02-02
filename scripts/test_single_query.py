#!/usr/bin/env python
import logging
import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.parser import H4ArgumentParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="mistral-nemo-12b-instruct")
    # llm = LLM(
    #     model=config.model_path,z
    #     gpu_memory_utilization=config.gpu_memory_utilization,
    #     enable_prefix_caching=True,
    #     seed=config.seed,
    #     tensor_parallel_size=num_gpus,
    # )
    prm = load_prm(config)

    # query = input("Enter your prompt/query: ")
    # TODO switch to code
    # query = "Let f(x) be a continuous function defined on R^3 such that for any two points P and Q on the unit sphere centered at origin with R as midpoint of shorter arc connecting P and Q, then f(P) + f(Q) >= 2f(R). Find minimum value of ∫∫∫_B f(x,y,z) dx dy dz where B is unit ball at origin. A) 4π/3 B) 2π/3 C) 0 D) -4π/3"
    query = "write a python function to binary search a rotated list"
    # approach function expects a dictionary with "problem" key.
    # so give it a single-element list containing the query.
    examples = {"problem": [query]}
    
    # run test-time compute method 
    approach_results = approach_fn(examples, config, llm, prm)

    # Extract outputs, approach_results should be:
    # {
    #   "completions": list[list[str]],
    #   "pred": list[str],
    #   "completion_tokens": list[int],
    #   "scores": list[list[float]]
    # }
    # print(approach_results)
    import numpy as np
    for k, v in approach_results.items():
        # Convert v to a numpy array; if nesting is irregular,
        # specify dtype=object to avoid errors.
        arr = np.array(v, dtype=object)
        print(f"Field: {k}")
        print(f"Shape: {arr.shape}")
        print()
    
    # for single prompt, results are at index 0
    best_answer = approach_results["pred"][0]
    all_completions = approach_results["completions"][0]
    scores = approach_results["scores"][0]

    print("---- Results ----")
    print(f"Prompt: {query}")
    print("\nAll completions:")
    for idx, comp in enumerate(all_completions):
        print(f"Completion {idx}: {comp}\nScore: {scores[idx]}")
    print("\nBest Completion:")
    print(best_answer)
    print("----------------")

if __name__ == "__main__":
    main()
