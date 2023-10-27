import json
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial
import os

from models.openai import chatgpts, gpts
from models.llama import LlamaInterface

from tasks import get_task
from tools import call_tools
from tools.search import search_save
from datetime import datetime

def get_fewshot_prompt(promptpath, task=None, chatgpt_format=False):
    if len(promptpath) == 0:
        return [] if chatgpt_format else ""
    elif promptpath == "default" and task is not None:
        return task.get_prompt()
    if not chatgpt_format:
        with open(f"./prompts/{promptpath}.txt", "r") as fin:
            prompt = fin.read() 
        return prompt
    else:
        with open(f"./prompts/{promptpath}.json", "r") as fin:
            prompt = json.load(fin)
        return prompt

def prepare_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

def prune_thought(prompt):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    return prompt

def run(task, idxs, gpts, evaluate=True, alpaca_format=False, chatgpt_format=True, promptpath='', question_prefix=''):
    fewshot_prompt = get_fewshot_prompt(promptpath, task, chatgpt_format=chatgpt_format)
    questions = [question_prefix + task[idx] for idx in idxs]
    if not chatgpt_format:
        prompts = [fewshot_prompt + question + "\n" for question in questions]
    else:
        prompts = [fewshot_prompt + [{'role': 'user', 'content': question}] for question in questions]
    if alpaca_format:
        prompts = [prepare_prompt(q.rstrip()) for q in questions]

    rs, infos = {}, {}
        
    if not chatgpt_format:
        query_queries_pairs = gpts([prompt for prompt in prompts])
    else:
        query_queries_pairs = gpts(prompts, stop=None)
    for _ in range(5):
        bad_ids = [i for i, pair in enumerate(query_queries_pairs) if "Queries: " not in pair]
        if not bad_ids: break

        bad_prompts = [prompts[i] for i in bad_ids]
        bad_pairs = gpts(bad_prompts, stop=None)
        for i, pair in zip(bad_ids, bad_pairs):
            query_queries_pairs[i] = pair
            if _ == 4 and "Queries: " not in pair:
                query_queries_pairs[i] = "Queries: [failed]"

    query, queries, bad_ids, done_ids = [], [], [], []
    for i, q_qs in enumerate(query_queries_pairs):
        if q_qs[:9] != "Queries: ":
            bad_ids.append(i)
            queries.append([])
            continue
        else:
            qs = q_qs[9:]
            if len(qs.strip().split('\\n')) <=1:
                bad_ids.append(i)
                queries.append([])
            else:
                queries.append(qs.strip().split('\\n'))

    for i, idx in enumerate(idxs):
        if i in bad_ids:
            continue
        info={}
        info['Query'] = questions[i]
        info['Queries'] = queries[i]
        infos[idxs[i]] = info 
    return infos


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    args.add_argument('--alpaca_format', action='store_true')
    args.add_argument('--chatgpt_format', action='store_true')
    args.add_argument('--question_prefix', type=str, default='')

    args.add_argument('--modelpath', type=str, default='')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--promptpath', type=str, default='')

    args = args.parse_args("--task triviaqa --backend gpt-4 --promptpath default --evaluate --random --task_split dev --temperature 0  --task_end_index 300".split())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = get_task(args.task, args.task_split)
    os.environ["OPENAI_API_KEY"]="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"
    modelname = args.backend
    if args.backend == 'llama':
        pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
        modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}_{args.task_split}_{args.task_start_index}_{args.task_end_index}_{modelname}_{args.temperature}_{time_str}.json"
    print(outfilename)
    
    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index:args.task_end_index]

    if args.backend == "llama":
        print(args.modelpath, args.peftpath, args.add_lora)
        llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
        model = llama.generate_responses_from_llama
    elif args.chatgpt_format:
        model = partial(chatgpts, model=args.backend, temperature=args.temperature)
    else:
        model = partial(gpts, model=args.backend, temperature=args.temperature)
        
    args.question_prefix="Query: "
    infos = run(task, idxs, model, evaluate=args.evaluate, \
                    alpaca_format=args.alpaca_format, 
                    chatgpt_format=args.chatgpt_format,
                    promptpath=args.promptpath,
                    question_prefix=args.question_prefix)

    with open(outfilename, "w") as fout:
        json.dump(infos, fout, indent=2)

    search_save()
