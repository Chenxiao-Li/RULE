"""
RULE - TTR (Test-Time Rethinking) for DBP15K Dataset.
Uses Qwen2.5-VL to re-evaluate entity alignment candidates via image/name similarity.
"""

import argparse
import os
import json
import time
import re

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import NeighborGenerator, get_score, evaluate_alignment, save_results_to_excel

# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser(description="TTR for DBP15K")
parser.add_argument("--data_choice", default="DBP15K", type=str, choices=["DBP15K"])
parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--eta", type=float, default=0.0, help="Noise ratio")
parser.add_argument("--use_surface", type=int, default=0, help="Whether to use surface (name) features")
parser.add_argument("--threshold", type=float, default=0.2, help="Confidence threshold to skip rethinking")
parser.add_argument("--use_previous_result", type=int, default=0, help="Resume from previous result file")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--save_step", type=int, default=100, help="Save results every N entities")
args = parser.parse_args()

# ==================== Derived Settings ====================
use_name = bool(args.use_surface)

# Naming convention matches `get_candidate/get_candidate_DBP.py`
# Format: DNC_{eta}[_use_surface]
setting = f"DNC_{args.eta}"
if use_name:
    setting += "_use_surface"

cand_filename = f"{setting}_{args.data_split}.json"
cand_file_path = os.path.join("./candidate_json", args.data_choice, cand_filename)

data_file_path = os.path.join("./data", args.data_choice, args.data_split)
output_filename = os.path.join("./result", args.data_choice, f"{setting}_{args.data_split}.json")
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

MLLM_PATH = "/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
IMG_HEIGHT, IMG_WIDTH = 150, 200

# ==================== Data Loading ====================
ng = NeighborGenerator(cand_file=cand_file_path, data_file_path=data_file_path)

# ==================== MLLM Initialization ====================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MLLM_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto",
)
processor = AutoProcessor.from_pretrained(MLLM_PATH)


# ==================== Prompt Templates ====================

def build_prompt_base(main_entity, unique_candidates):
    """Build the shared context prompt for image/name scoring requests."""
    if use_name:
        base = ("Help me align or match entities of different knowledge graphs "
                "according to the given names, images and prior retrieval results."
                "\nBelow are prior retrieval results focusing on visual and textual similarity "
                "of the given images and names, respectively.")
        cand_list = ', '.join(f"{c['ent_id']} {c['name']} {c['hhea_sim']:.2f}" for c in unique_candidates)
        fmt = "ID Name Similarity"
    else:
        base = ("Help me align or match entities of different knowledge graphs "
                "according to the given images and prior retrieval results."
                "\nBelow are prior retrieval results focusing on visual similarity of the given images.")
        cand_list = ', '.join(f"{c['ent_id']} {c['hhea_sim']:.2f}" for c in unique_candidates)
        fmt = "ID Similarity"

    main_info = f"ID:{main_entity['ent_id']}" + (f" Name:{main_entity['name']}" if use_name else "")
    base += (f"\n[Candidate Entities List] which may be aligned with QUERY Entity ({main_info}) "
             f"are shown in the following list [Format: {fmt}]: [{cand_list}].")
    return base


def build_image_prompt(main_entity, candidate):
    """Build the per-candidate image comparison prompt."""
    q_info = f"ID:{main_entity['ent_id']}" + (f" Name:{main_entity['name']}" if use_name else "")
    c_info = f"ID:{candidate['ent_id']}" + (f" Name:{candidate['name']}" if use_name else "")
    return (
        f"The two provided images represent the query ({q_info}) and the candidate ({c_info}).\n"
        "Please evaluate the probability that the QUERY and the CANDIDATE belong to the same entity STEP BY STEP:\n"
        "1. Rethink the visual similarities based on the prior retrieval results and the given images.\n"
        "2. Analyze the similarities of detailed visual contents between the provided images.\n"
        "3. Consider the underlying connections between the given images.\n"
        "[Output Format]: [IMAGE SIMILARITY] = A out of 10, where A is in range [0,1,2,3,4,5,6,7,8,9,10], "
        "which represents the levels from VERY LOW to VERY HIGH.\n"
        "NOTICE: You MUST output strictly in this format: [IMAGE SIMILARITY] = A out of 10."
    )


def build_name_prompt(main_entity, candidate):
    """Build the per-candidate name comparison prompt."""
    return (
        f"The two provided names represent the query (ID:{main_entity['ent_id']} Name:{main_entity['name']}) "
        f"and the candidate (ID:{candidate['ent_id']} Name:{candidate['name']}), respectively.\n"
        "Based on the prior retrieval results and the given names, identify the similarities "
        "between the query entity and candidate entity.\n"
        "[Output Format]: [NAME SIMILARITY] = B out of 10, where B ∈ {0,1,2,3,4,5,6,7,8,9,10} "
        "represent the levels from VERY LOW to VERY HIGH.\n"
        "NOTICE: You MUST output strictly in this format: [NAME SIMILARITY] = B out of 10."
    )


# 【新增】为Qwen原始分数相同的候选实体构造联合tie-breaking Prompt
def build_tie_prompt(main_entity, tied_candidates, raw_score):
    """Build a joint tie-breaking prompt for candidates with the same Qwen score."""
    q_info = f"ID:{main_entity['ent_id']}" + (f" Name:{main_entity['name']}" if use_name else "")
    cand_text = "\n".join(
        f"CANDIDATE_{i + 1}: ID={c['ent_id']}" + (f" Name={c['name']}" if use_name else "")
        for i, c in enumerate(tied_candidates)
    )
    return (
        f"The following candidates received the same similarity score ({raw_score}) for the query ({q_info}).\n"
        "Compare ONLY these tied candidates jointly and break the tie. Rank them from most likely to least likely "
        "to represent the same real-world entity as the query. Do not output tied ranks. Every candidate must appear "
        "exactly once.\n\n"
        f"{cand_text}\n\n"
        "Output exactly one final line in this format:\n"
        "[RANKING] = CANDIDATE_1 > CANDIDATE_2 > ..."
    )


# 【新增】解析Qwen返回的同分候选排序，保证每个候选只出现一次
def parse_tie_ranking(response, num_candidates):
    labels = re.findall(r"CANDIDATE_\d+", response)
    ranking = []

    for label in labels:
        index = int(label.split("_")[-1])
        if 1 <= index <= num_candidates and label not in ranking:
            ranking.append(label)

    for i in range(1, num_candidates + 1):
        label = f"CANDIDATE_{i}"
        if label not in ranking:
            ranking.append(label)

    return ranking[:num_candidates]


# ==================== MLLM Inference ====================

def batch_inference(requests, max_new_tokens=384):
    """Send a batch of image/name requests to Qwen2.5-VL and return decoded responses."""
    processor.tokenizer.padding_side = "left"
    messages = []
    for req in requests:
        if 'image_prompt' in req:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": req['main_entity_img'], "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                    {"type": "image", "image": req['candidate_entity_img'], "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                    {"type": "text", "text": req['image_prompt']},
                ]},
            ])
        elif 'name_prompt' in req:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req['name_prompt']},
            ])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# 【新增】对同分候选额外调用一次Qwen进行联合比较，得到同分组内部唯一顺序
def break_tie_with_llm(main_entity, tied_candidates, raw_score):
    """Use one extra joint Qwen inference to order candidates with the same score."""
    if len(tied_candidates) <= 1:
        return tied_candidates

    main_img = main_entity.get('img_path', '')
    if not (main_img and os.path.exists(main_img)):
        return tied_candidates

    valid_candidates = []
    content = [
        {"type": "image", "image": main_img, "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH}
    ]

    for candidate in tied_candidates:
        cand_img = candidate.get('img_path', '')
        if cand_img and os.path.exists(cand_img):
            valid_candidates.append(candidate)
            content.append({
                "type": "image",
                "image": cand_img,
                "resized_height": IMG_HEIGHT,
                "resized_width": IMG_WIDTH,
            })

    if len(valid_candidates) <= 1:
        return tied_candidates

    content.append({
        "type": "text",
        "text": build_tie_prompt(main_entity, valid_candidates, raw_score)
    })

    messages = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]]

    text_prompt = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    ranking_labels = parse_tie_ranking(response, len(valid_candidates))
    ranked_valid = [valid_candidates[int(label.split("_")[-1]) - 1] for label in ranking_labels]
    missing_candidates = [c for c in tied_candidates if c not in valid_candidates]

    del inputs
    del generated_ids
    del trimmed
    del image_inputs
    del video_inputs
    torch.cuda.empty_cache()

    return ranked_valid + missing_candidates


# ==================== Qwen-only Score Ranking ====================

# 【修改】这里只保留原始Qwen Image/Name Score，不在这里人工拆分同分
def fuse_scores(image_scores, name_scores, unique_candidates):
    """Use only Qwen image/name scores as raw candidate scores."""
    candidate_ids = [c['ent_id'] for c in unique_candidates]

    image_tensor = torch.tensor(
        [image_scores.get(cid, 0) for cid in candidate_ids],
        dtype=torch.float32,
    )

    if use_name:
        name_tensor = torch.tensor(
            [name_scores.get(cid, 0) for cid in candidate_ids],
            dtype=torch.float32,
        )
        final_scores = image_tensor + name_tensor
    else:
        final_scores = image_tensor

    # 【新增】保存每个候选的原始LLM分数，后面用它检测哪些候选出现同分
    raw_scores = {
        cid: final_scores[candidate_ids.index(cid)].item()
        for cid in candidate_ids
    }

    return final_scores, candidate_ids, raw_scores


# ==================== Single Entity Evaluation ====================

def eval_single_entity(main_entity, candidate_entities, ref_ent, base_rank):
    """Evaluate alignment for one query entity. Returns (rank, time_cost, details)."""
    st = time.time()
    rank = base_rank
    empty_result = (rank, time.time() - st, [], False)

    # Skip if base rank is already out of top-10
    if base_rank >= 10:
        return empty_result

    # Deduplicate candidates
    unique_cands = list({c['ent_id']: c for c in candidate_entities if isinstance(c, dict)}.values())

    # Early exit if retrieval is already confident
    ori_scores = sorted([c.get("hhea_sim", 0) for c in unique_cands], reverse=True)
    if ori_scores[0] >= args.threshold or (len(ori_scores) > 1 and ori_scores[0] - ori_scores[1] > 0.2):
        return empty_result

    prompt_base = build_prompt_base(main_entity, unique_cands)
    image_scores, name_scores = {}, {}

    # --- Build & send image requests ---
    main_img = main_entity.get('img_path', '')
    if not (main_img and os.path.exists(main_img)):
        main_img = ''

    image_requests, img_mapping = [], {}
    if not main_img:
        for c in unique_cands:
            image_scores[c['ent_id']] = 0
    else:
        for c in unique_cands:
            cand_img = c.get('img_path', '')
            if cand_img and os.path.exists(cand_img):
                img_mapping[c['ent_id']] = len(image_requests)
                image_requests.append({
                    'main_entity_img': main_img,
                    'candidate_entity_img': cand_img,
                    'image_prompt': prompt_base + "\n" + build_image_prompt(main_entity, c),
                })
            else:
                image_scores[c['ent_id']] = 0

    if image_requests:
        img_responses = batch_inference(image_requests)
        for eid, idx in img_mapping.items():
            image_scores[eid] = get_score(img_responses[idx])

    # --- Build & send name requests ---
    name_mapping = {}
    if use_name:
        name_requests = []
        for c in unique_cands:
            name_mapping[c['ent_id']] = len(name_requests)
            name_requests.append({'name_prompt': prompt_base + "\n" + build_name_prompt(main_entity, c)})
        
        name_responses = batch_inference(name_requests)
        for eid, idx in name_mapping.items():
            name_scores[eid] = get_score(name_responses[idx])

    # 【修改】先得到Qwen原始分数，此时允许多个候选实体同分
    final_scores, candidate_ids, raw_scores = fuse_scores(image_scores, name_scores, unique_cands)

    # 【新增】按照Qwen原始分数分组，找出所有同分候选
    score_groups = {}
    for c in unique_cands:
        score = raw_scores[c['ent_id']]
        score_groups.setdefault(score, []).append(c)

    # 【新增】只有同分组才额外调用一次Qwen；不同分的候选不增加推理
    sorted_cands = []
    tie_break_details = []
    for score in sorted(score_groups.keys(), reverse=True):
        group = score_groups[score]

        if len(group) > 1:
            original_ids = [c['ent_id'] for c in group]
            group = break_tie_with_llm(main_entity, group, score)
            tie_break_details.append({
                "raw_score": score,
                "before": original_ids,
                "after": [c['ent_id'] for c in group],
            })

        sorted_cands.extend(group)

    # 【新增】根据最终完整排序强制赋唯一分数：第1名=9，第2名=8，...，第10名=0
    unique_scores = {
        c['ent_id']: 9 - rank_index
        for rank_index, c in enumerate(sorted_cands)
    }

    # --- Build detail list ---
    details = []
    for c in candidate_entities:
        d = {
            "ent_id": c['ent_id'],
            "image_score": image_scores.get(c['ent_id'], 0),
            # 【新增】记录tie-breaking之前的原始Qwen分数
            "raw_llm_score": raw_scores.get(c['ent_id'], 0),
            # 【新增】记录tie-breaking之后按最终排名得到的唯一分数
            "unique_score": unique_scores.get(c['ent_id'], 0),
            "ori_score": c.get("hhea_sim", 0),
        }
        if use_name:
            d["name_score"] = name_scores.get(c['ent_id'], 0)
        details.append(d)

    # --- Determine final rank ---
    for j, c in enumerate(sorted_cands):
        if c['ent_id'] == ref_ent:
            rank = j
            break

    return rank, time.time() - st, details, True


# ==================== Main Evaluation Loop ====================

def run_evaluation(hit_k=[1, 5, 10]):
    """Run TTR evaluation over all entities."""
    base_ranks, llm_ranks = [], []
    main_entities = ng.get_entities()
    result, processed_ids = {}, set()
    improved_to_top1 = []
    improved_not_top1 = []
    ranking_worse = []
    used_llm_count = 0
    used_llm_ids = []

    # Resume from previous run
    if args.use_previous_result and os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} existing results from {output_filename}")
        for eid, res in result.items():
            base_ranks.append(res["base_rank"])
            llm_ranks.append(res["llm_rank"])
            processed_ids.add(eid)
        main_entities = [e for e in main_entities if str(e) not in processed_ids]

    total = len(main_entities)
    for i, ent_id in enumerate(tqdm(main_entities, desc="Reasoning & Rethinking"), 1):
        candidates = ng.get_candidates(ent_id)
        ref_ent = ng.get_ref_ent(ent_id)
        base_rank = ng.get_base_rank(ent_id)
        main_ent = ng.get_main_entity(ent_id)

        llm_rank, cost, details, used_llm = eval_single_entity(main_ent, candidates, ref_ent, base_rank)
        
        if used_llm:
            used_llm_count += 1
            used_llm_ids.append(ent_id)
                
        if base_rank != 0 and llm_rank == 0:
            improved_to_top1.append((ent_id, base_rank + 1, llm_rank + 1))
        
        elif llm_rank < base_rank:
            improved_not_top1.append((ent_id, base_rank + 1, llm_rank + 1))
        
        elif llm_rank > base_rank:
            ranking_worse.append((ent_id, base_rank + 1, llm_rank + 1))

        base_ranks.append(base_rank)
        llm_ranks.append(llm_rank)
        result[ent_id] = {
            "base_rank": int(base_rank), "llm_rank": int(llm_rank),
            "candidates": details, "time_cost": cost,
        }

        # Periodic checkpoint
        if i % args.save_step == 0 or i == total:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            base_hits, base_mrr = evaluate_alignment(base_ranks, hit_k)
            llm_hits, llm_mrr = evaluate_alignment(llm_ranks, hit_k)
            print(f"\n[{i}/{total}] Base: Hits@{hit_k}={base_hits}, MRR={base_mrr:.3f}")
            print(f"[{i}/{total}]  TTR: Hits@{hit_k}={llm_hits}, MRR={llm_mrr:.3f}")

            if i == total:
                save_results_to_excel(args, {"metrics": {
                    "base_rank": {"Hits@1": base_hits[0], "Hits@5": base_hits[1], "Hits@10": base_hits[2], "MRR": base_mrr},
                    "llm_rank": {"Hits@1": llm_hits[0], "Hits@5": llm_hits[1], "Hits@10": llm_hits[2], "MRR": llm_mrr},
                }})

    print("\n" + "=" * 100)
    print("Improved to Rank-1")
    print("=" * 100)

    for ent_id, old_rank, new_rank in improved_to_top1:
        print(f"{ent_id}: {old_rank} -> {new_rank}")

    print(f"Total: {len(improved_to_top1)}")


    print("\n" + "=" * 100)
    print("Improved but NOT Rank-1")
    print("=" * 100)

    for ent_id, old_rank, new_rank in improved_not_top1:
        print(f"{ent_id}: {old_rank} -> {new_rank}")

    print(f"Total: {len(improved_not_top1)}")


    print("\n" + "=" * 100)
    print("Ranking Worse")
    print("=" * 100)

    for ent_id, old_rank, new_rank in ranking_worse:
        print(f"{ent_id}: {old_rank} -> {new_rank}")

    print(f"Total: {len(ranking_worse)}")
    print("\n" + "=" * 100)
    print("Entities Using LLM Reranking")
    print("=" * 100)
    print(f"Total: {used_llm_count}/{len(ng.get_entities())}")
    print(f"Entity IDs: {used_llm_ids}")
    
    return result


# ==================== Entry Point ====================

if __name__ == '__main__':
    start = time.time()
    result = run_evaluation(hit_k=[1, 5, 10])
    print(f"Total time: {time.time() - start:.2f}s")