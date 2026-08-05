"""Run TTR reranking for DBP15K entity ID 4504 only."""

import argparse
import os
import time

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import NeighborGenerator, get_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_choice", default="DBP15K")
parser.add_argument("--data_split", default="zh_en", choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--use_surface", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--target_ids", nargs="+", type=int, default=[24135, 25148])
args = parser.parse_args()

use_name = bool(args.use_surface)
setting = f"DNC_{args.eta}" + ("_use_surface" if use_name else "")
cand_file_path = os.path.join("./candidate_json", args.data_choice, f"{setting}_{args.data_split}.json")
data_file_path = os.path.join("./data", args.data_choice, args.data_split)

MLLM_PATH = "/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
IMG_HEIGHT, IMG_WIDTH = 150, 200

ng = NeighborGenerator(cand_file=cand_file_path, data_file_path=data_file_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MLLM_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MLLM_PATH)


def build_prompt_base(main_entity, candidates):
    if use_name:
        base = ("Help me align or match entities of different knowledge graphs according to the given names, "
                "images and prior retrieval results.\nBelow are prior retrieval results focusing on visual and "
                "textual similarity of the given images and names, respectively.")
        cand_list = ", ".join(f"{c['ent_id']} {c['name']} {c['hhea_sim']:.2f}" for c in candidates)
        fmt = "ID Name Similarity"
    else:
        base = ("Help me align or match entities of different knowledge graphs according to the given images "
                "and prior retrieval results.\nBelow are prior retrieval results focusing on visual similarity "
                "of the given images.")
        cand_list = ", ".join(f"{c['ent_id']} {c['hhea_sim']:.2f}" for c in candidates)
        fmt = "ID Similarity"

    main_info = f"ID:{main_entity['ent_id']}" + (f" Name:{main_entity['name']}" if use_name else "")
    return base + (f"\n[Candidate Entities List] which may be aligned with QUERY Entity ({main_info}) "
                   f"are shown in the following list [Format: {fmt}]: [{cand_list}].")


def build_image_prompt(main_entity, candidate):
    q_info = f"ID:{main_entity['ent_id']}" + (f" Name:{main_entity['name']}" if use_name else "")
    c_info = f"ID:{candidate['ent_id']}" + (f" Name:{candidate['name']}" if use_name else "")
    return (
        f"The two provided images represent the query ({q_info}) and the candidate ({c_info}).\n"
        "Please evaluate the probability that the QUERY and the CANDIDATE belong to the same entity STEP BY STEP:\n"
        "1. Rethink the visual similarities based on the prior retrieval results and the given images.\n"
        "2. Analyze the similarities of detailed visual contents between the provided images.\n"
        "3. Consider the underlying connections between the given images.\n"
        "[Output Format]: [IMAGE SIMILARITY] = A out of 10, where A is in range [0,1,2,3,4,5,6,7,8,9,10].\n"
        "NOTICE: You MUST output strictly in this format: [IMAGE SIMILARITY] = A out of 10."
    )


def build_name_prompt(main_entity, candidate):
    return (
        f"The two provided names represent the query (ID:{main_entity['ent_id']} Name:{main_entity['name']}) "
        f"and the candidate (ID:{candidate['ent_id']} Name:{candidate['name']}), respectively.\n"
        "Based on the prior retrieval results and the given names, identify their similarity.\n"
        "[Output Format]: [NAME SIMILARITY] = B out of 10, where B is in range [0,1,2,3,4,5,6,7,8,9,10].\n"
        "NOTICE: You MUST output strictly in this format: [NAME SIMILARITY] = B out of 10."
    )


def batch_inference(requests, max_new_tokens=384):
    processor.tokenizer.padding_side = "left"
    messages = []
    for req in requests:
        if "image_prompt" in req:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": req["main_entity_img"], "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                    {"type": "image", "image": req["candidate_entity_img"], "resized_height": IMG_HEIGHT, "resized_width": IMG_WIDTH},
                    {"type": "text", "text": req["image_prompt"]}
                ]}
            ])
        else:
            messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req["name_prompt"]}
            ])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def fuse_scores(image_scores, name_scores, candidates):
    """
    Rank candidates using only Qwen scores.

    use_surface = 0:
        final_score = image_score

    use_surface = 1:
        final_score = image_score + name_score
    """
    candidate_ids = [c["ent_id"] for c in candidates]

    image_tensor = torch.tensor(
        [image_scores.get(candidate_id, 0) for candidate_id in candidate_ids],
        dtype=torch.float32,
    )

    if use_name:
        name_tensor = torch.tensor(
            [name_scores.get(candidate_id, 0) for candidate_id in candidate_ids],
            dtype=torch.float32,
        )
        final_scores = image_tensor + name_tensor
    else:
        final_scores = image_tensor

    return final_scores


def run_single_entity(ent_id):
    if ent_id not in ng.get_entities():
        print(f"\nEntity {ent_id} is not found in {cand_file_path}")
        return None

    main_entity = ng.get_main_entity(ent_id)
    candidates = ng.get_candidates(ent_id)
    ref_ent = ng.get_ref_ent(ent_id)
    base_rank = ng.get_base_rank(ent_id)
    unique_candidates = list({c["ent_id"]: c for c in candidates}.values())

    print("\n" + "=" * 80)
    print(f"Query ID: {ent_id}")
    print(f"Query name: {main_entity['name']}")
    print(f"Reference ID: {ref_ent}")
    print(f"Base rank: {base_rank} ({base_rank + 1}th position)")
    print(f"Candidates: {[c['ent_id'] for c in unique_candidates]}")
    if use_name:
        print("Ranking rule: final_score = Qwen image_score + Qwen name_score")
    else:
        print("Ranking rule: final_score = Qwen image_score")

    if base_rank >= 10:
        print("Reference entity is outside Top-10; TTR is skipped.")
        return {"ent_id": ent_id, "base_rank": base_rank, "llm_rank": base_rank, "correct": False, "skipped": True}

    ori_scores = sorted([c["hhea_sim"] for c in unique_candidates], reverse=True)
    if ori_scores[0] >= args.threshold or (len(ori_scores) > 1 and ori_scores[0] - ori_scores[1] > 0.2):
        print("TTR is skipped by the confidence rule.")
        print(f"Final rank: {base_rank} ({base_rank + 1}th position)")
        print(f"Hits@1 correct: {base_rank == 0}")
        return {"ent_id": ent_id, "base_rank": base_rank, "llm_rank": base_rank, "correct": base_rank == 0, "skipped": True}

    start = time.time()
    prompt_base = build_prompt_base(main_entity, unique_candidates)
    image_scores, name_scores = {}, {}

    main_img = main_entity["img_path"] if os.path.exists(main_entity["img_path"]) else ""
    image_requests, image_ids = [], []

    for candidate in unique_candidates:
        cand_img = candidate["img_path"]
        if main_img and os.path.exists(cand_img):
            image_ids.append(candidate["ent_id"])
            image_requests.append({
                "main_entity_img": main_img,
                "candidate_entity_img": cand_img,
                "image_prompt": prompt_base + "\n" + build_image_prompt(main_entity, candidate)
            })
        else:
            image_scores[candidate["ent_id"]] = 0

    if image_requests:
        responses = batch_inference(image_requests)
        for candidate_id, response in zip(image_ids, responses):
            image_scores[candidate_id] = get_score(response)
            print(f"\nImage response for {candidate_id}:\n{response}")

    if use_name:
        name_requests = [{"name_prompt": prompt_base + "\n" + build_name_prompt(main_entity, c)} for c in unique_candidates]
        responses = batch_inference(name_requests)
        for candidate, response in zip(unique_candidates, responses):
            name_scores[candidate["ent_id"]] = get_score(response)
            print(f"\nName response for {candidate['ent_id']}:\n{response}")

    final_scores = fuse_scores(image_scores, name_scores, unique_candidates)
    ranked = sorted(zip(unique_candidates, final_scores.tolist()), key=lambda x: x[1], reverse=True)
    llm_rank = next(i for i, (candidate, _) in enumerate(ranked) if candidate["ent_id"] == ref_ent)

    print("\nQwen-only reranking result")
    for rank, (candidate, score) in enumerate(ranked, 1):
        mark = "  <-- ground truth" if candidate["ent_id"] == ref_ent else ""
        name = f" | {candidate['name']}" if use_name else ""
        print(f"{rank:2d}. ID={candidate['ent_id']}{name} | ori={candidate['hhea_sim']:.3f} "
              f"| image={image_scores.get(candidate['ent_id'], 0)}"
              + (f" | name={name_scores.get(candidate['ent_id'], 0)}" if use_name else "")
              + f" | final={score:.6f}{mark}")

    print(f"\nBase rank: {base_rank} ({base_rank + 1}th position)")
    print(f"LLM rank: {llm_rank} ({llm_rank + 1}th position)")
    print(f"Rank improved: {llm_rank < base_rank}")
    print(f"Hits@1 correct: {llm_rank == 0}")
    print(f"Time cost: {time.time() - start:.2f}s")

    return {
        "ent_id": ent_id,
        "base_rank": base_rank,
        "llm_rank": llm_rank,
        "correct": llm_rank == 0,
        "improved": llm_rank < base_rank,
        "skipped": False
    }


def main():
    results = []
    total_start = time.time()

    for ent_id in args.target_ids:
        result = run_single_entity(ent_id)
        if result is not None:
            results.append(result)

    print("\n" + "=" * 80)
    print("Qwen-only Reranking Summary")
    print(f"Total entities: {len(results)}")
    print(f"Hits@1 correct: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Improved: {sum(r.get('improved', False) for r in results)}/{len(results)}")
    print(f"Skipped: {sum(r['skipped'] for r in results)}/{len(results)}")

    for r in results:
        print(f"ID={r['ent_id']} | base_rank={r['base_rank']} | llm_rank={r['llm_rank']} "
              f"| correct={r['correct']} | skipped={r['skipped']}")

    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
