#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm: PRM):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=2048,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    best_scores=[0.0],
                    all_scores=[],
                    previous_text=None,
                    pruned=False,
                    stop_reasons=None,
                    history=[],
                )
            )

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        if i == config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=2048,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in gen_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        # tokenizer = llm.get_tokenizer()
        # # TODO: set the augmented template from a file
        # if config.custom_chat_template is not None:
        #     tokenizer.chat_template = config.custom_chat_template
        # templated_convs = tokenizer.apply_chat_template(
        #     convs,
        #     add_generation_prompt=add_generation_prompt,
        #     continue_final_message=continue_final_message,
        #     tokenize=False,
        # )
        templated_convs = [
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"Cutting Knowledge Date: December 2023\n"
            f"Today Date: 09 Jan 2025\n\n"
            f"Solve the following math problem efficiently and clearly:\n\n"
            f"- For simple problems (2 steps or fewer):\n"
            f"Provide a concise solution with minimal explanation.\n\n"
            f"- For complex problems (3 steps or more):\n"
            f"Use this step-by-step format:\n\n"
            f"## Step 1: [Concise description]\n"
            f"[Brief explanation and calculations]\n\n"
            f"## Step 2: [Concise description]\n"
            f"[Brief explanation and calculations]\n\n"
            f"...\n\n"
            f"Regardless of the approach, always conclude with:\n\n"
            f"Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\n"
            f"Where [answer] is just the final number or expression that solves the problem."
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f'      "prompt": "{conv[1]["content"]}",'
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            for conv in convs
        ]

        # print(f"####################################\nTEMPLATED CONVS: {templated_convs}")
        # print(f"####################################\nTEMPLATED CONVS 2: {templated_convs2}")

        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, config.beam_width
        )

        prompts, completions = [], []
        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            if len(beam.next_texts) != config.beam_width:
                beam.pruned = True
                # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )
            prompts.append(beam.prompt)
            completions.append([beam.current_text + t for t in beam.lookahead_texts])

        # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt

        all_scores = prm.score(prompts, completions)

        for beam, scores in zip(gen_beams, all_scores, strict=True):
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores]
            best_score_ind = np.argmax(agg_scores)
            beam.all_scores = scores
            beam.previous_text = beam.current_text
            beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            beam.best_scores = scores[best_score_ind]
            if (
                beam.next_texts[best_score_ind] == ""
                or beam.stop_reasons[best_score_ind] == "EOS"
            ):
                # stopped on EOS, prune
                beam.pruned = True

        # filter / prune
        for beam in gen_beams:
            if "boxed{" in beam.current_text:
                beam.pruned = True

    # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            output.append(
                Beam(
                    prompt=beam.prompt,
                    index=beam.index,
                    current_text=beam.previous_text + beam.next_texts[i],
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=beam.all_scores[i],
                    all_scores=beam.all_scores,
                    previous_text=beam.current_text,
                    pruned=beam.pruned,
                    history=beam.history,
                )
            )

    return output


def dvts(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _dvts(problems, config, llm, prm)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(b.best_scores, config.agg_strategy)
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append(-1)

    # TODO: construct and store the tree

    return results
