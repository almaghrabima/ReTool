# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

import os
import re
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm, trange
from datasets import load_from_disk, load_dataset
from evaluator.MC_evaluator_list import MCEvaluator
from evaluator.MATH_evaluator_list import MATHEvaluator
from executor import *

def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness

name2path = {
    "AIME24": "dataset/AIME24.jsonl",
    "AIME25": "dataset/AIME25.jsonl",
}

name2eval = {
    "AIME24": MATHEvaluator(),
    "AIME25": MATHEvaluator(),
}


def main(args, lines, start_id, use_slice=False):
    import os

    if use_slice:
        # adjusted based on the number of slices
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            f"{start_id%2*4},{start_id%2*4+1},{start_id%2*4+2},{start_id%2*4+3}"
        )

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.paralle_size,
        swap_space=16,
    )
    stop_words = []
    
    if args.exe_code:
        stop_words.append("</code>")
    executor = PythonExecutor()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=args.n,
    )
    sampling_params_1 = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=1,
    )
    evaluator = name2eval[args.data_name]

    def excute_codes(codes, executor: PythonExecutor):
        no_code_idx = []
        codes_use = []
        for i, code in enumerate(codes):
            if code == "":
                no_code_idx.append(i)
            else:
                codes_use.append(code)
        batch_results = executor.batch_apply(codes_use)
        return batch_results, no_code_idx

    def process_prompt(question):
        with open(args.prompt_template, "r") as fin:
            sys = json.load(fin)
        prompt_prefix = sys[args.prompt]
        chat_prob = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_prefix.format(query=question),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return chat_prob

    if args.exe_code:
        prefix_tgt = "exe"
    else:
        prefix_tgt = "no_exe"

    tgt_path = os.path.join(
        args.target_path,
        "{}-{}-{}-{}-{}.jsonl".format(
            prefix_tgt,
            args.model_name_or_path.split("/")[-1],
            args.data_name.split("/")[-1],
            args.prompt_template.split("/")[-1].split(".")[0],
            args.n,
        ),
    )
    fout = open(tgt_path, "w")

    bs = 100
    num_data = len(lines)
    total_problem, total_correct = 0, 0
    finished_cnt = 0
    for st in trange(0, num_data, bs):
        print(
            "start_id: {}, st: {}, bs: {}, num_data: {}".format(
                start_id, st, bs, num_data
            )
        )
        tmp_lines = lines[st : st + bs]

        # when ouput code tokens, we need to stop, use code interpreter to run the code, and then insert the output to the input, and continue to generate the next part of the code
        prompts = [process_prompt(data["input"]) for data in tmp_lines]
        responses = model.generate(prompts, sampling_params)
        final_responses = []
        final_code_num_lst = []
        response_idx = 0
        for response, prompt in zip(responses, prompts):
            response_idx += 1
            print("===" * 9)
            print("processing st: {}, response_idx: {}".format(st, response_idx))
            print("===" * 9)
            code_num_lst = [0 for _ in range(len(response.outputs))]
            intermediate_responses = [prompt for _ in range(len(response.outputs))]
            fini_responses = []
            pred_stop_reason_lst = [
                [output.text, output.stop_reason] for output in response.outputs
            ]
            while any(
                [
                    pred_stop_reason is not None
                    for pred_stop_reason in pred_stop_reason_lst
                ]
            ):
                code_to_execute_lst = []
                assert len(pred_stop_reason_lst) == len(intermediate_responses)
                for res_idx in range(len(pred_stop_reason_lst)):
                    pred_stop_reason = pred_stop_reason_lst[res_idx]
                    inter_response = intermediate_responses[res_idx]
                    if inter_response is None:
                        continue
                    pred, stop_reason = pred_stop_reason
                    if stop_reason != "</code>":
                        fini_responses.append(inter_response + pred)
                        pred_stop_reason_lst[res_idx] = None
                        intermediate_responses[res_idx] = None
                        continue
                    else:
                        code_to_execute_lst.append(pred.split("```python")[-1].replace("```", "").strip())
                        intermediate_responses[res_idx] = inter_response + pred
                if len(code_to_execute_lst) == 0:
                    break
                batch_results, no_code_idx = excute_codes(
                    code_to_execute_lst, executor=executor
                )
                batch_results_include_none = []
                for i in range(len(code_to_execute_lst)):
                    if i in no_code_idx:
                        batch_results_include_none.append(None)
                    else:
                        batch_results_include_none.append(batch_results.pop(0))
                for i, inter_response in enumerate(intermediate_responses):
                    if inter_response is None:
                        continue
                    exe_result = batch_results_include_none.pop(0)
                    if exe_result is None:
                        excu_content = "None"
                    else:
                        output, report = exe_result
                        if report == "Done":
                            excu_content = output
                        else:
                            excu_content = report

                    intermediate_responses[i] += (
                        "</code>\n" + "<interpreter>\n" + excu_content + "</interpreter>\n\n"
                    )

                intermediate_responses_to_gen = [
                    inter_response
                    for inter_response in intermediate_responses
                    if inter_response is not None
                ]
                new_intermediate_responses = model.generate(
                    intermediate_responses_to_gen, sampling_params_1
                )
                tmp_cnt = 0
                for new_i, pred_stop_reason in enumerate(pred_stop_reason_lst):
                    if pred_stop_reason is not None:
                        tmp_output = new_intermediate_responses[tmp_cnt].outputs.pop(0)
                        tmp_cnt += 1
                        pred_stop_reason_lst[new_i] = [
                            tmp_output.text,
                            tmp_output.stop_reason,
                        ]
                        code_num_lst[new_i] += 1

            final_responses.append(fini_responses)
            final_code_num_lst.append(code_num_lst)

        for response, data, code_num_lst in zip(
            final_responses, tmp_lines, final_code_num_lst
        ):
            output_ = data["output"]
            new_data = {
                "input": data["input"],
                "output": output_,
                "prediction": [],
            }
            pred_ans_list, real_ans_list = [], []
            pred_ans_list_rm_think = []
            for pred in response:
                pred_ans_list.append(pred)
                real_ans_list.append(output_)
                pred_ans_list_rm_think.append(pred.split("</think>")[-1].strip())

            correctness = check(evaluator, pred_ans_list, real_ans_list)

            pred_last_num_lst = [
                re.findall(r"\d+", pred_ans.split("\n")[-1])
                for pred_ans in pred_ans_list
            ]
            pred_real_pairs = [
                (
                    (False, real_ans)
                    if len(pred_last_num) == 0
                    else ("\\boxed{" + pred_last_num[-1] + "}", real_ans)
                )
                for pred_last_num, real_ans in zip(pred_last_num_lst, real_ans_list)
            ]
            correctness_last_num_left = check(
                evaluator,
                [c[0] for c in pred_real_pairs if c[0] != False],
                [c[1] for c in pred_real_pairs if c[0] != False],
            )
            correctness_last_num = []
            for idx in range(len(pred_real_pairs)):
                if pred_real_pairs[idx][0] == False:
                    correctness_last_num.append(False)
                else:
                    correctness_last_num.append(correctness_last_num_left.pop(0))
            correctness = [
                c or c_last_num
                for c, c_last_num in zip(correctness, correctness_last_num)
            ]


            cnt = 0
            for output, c, code_num in zip(response, correctness, code_num_lst):
                pred = output
                stop_reason = None
                if c is True:
                    total_correct = total_correct + 1
                token_len = len(tokenizer.encode(pred))
                new_data["prediction"].append(
                    {
                        "solution": pred,
                        "correctness": c,
                        "stop_reason": stop_reason,
                        "token_len": token_len,
                        "code_num": code_num,
                    }
                )
                total_problem = total_problem + 1
            fout.write(json.dumps(new_data) + "\n")
            fout.flush()

    results = {
        "accuracy": round(total_correct / total_problem * 100, 2)
    }
    fout.write(json.dumps(results) + "\n")
    fout.flush()

    fout.close()
    print(
        "Accuracy: {}: {}% ( {} / {} )".format(
            args.data_name.split("/")[-1],
            round(total_correct / total_problem * 100, 2),
            total_correct,
            total_problem,
        )
    )
    print("===" * 9)
    print(args.model_name_or_path)
    print("===" * 9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_tokens", default=10000, type=int)
    parser.add_argument("--paralle_size", default=8, type=int)
    parser.add_argument("--year", default=None, type=str, required=False)
    parser.add_argument("--prompt", default="r1_code", type=str, required=False)
    parser.add_argument("--decode", default="sample", type=str)
    parser.add_argument("--use_slice", action="store_true")
    parser.add_argument("--slice_id", default=0, type=int)
    parser.add_argument("--prompt_template", default=None, type=str)
    parser.add_argument("--n", default=8, type=int)
    parser.add_argument("--exe_code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)

    src_path = name2path[args.data_name]
    with open(src_path, "r") as fin:
        raw_dataset = fin.readlines()
        dataset = [json.loads(d) for d in raw_dataset]
    print("Total data: {}".format(len(dataset)))

    if args.use_slice:
        slice_idx = np.linspace(0, len(dataset), 3).astype("int")
        start, end = slice_idx[args.slice_id], slice_idx[args.slice_id + 1]
        dataset = dataset[start:end]
        print(f"start process {args.slice_id} from {start} to {end}")

    main(args, dataset, args.slice_id, args.use_slice)
