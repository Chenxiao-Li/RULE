"""
DBP15K 批量局部视觉分割:
Entity Semantic Info (+ optional Entity Name) + Image
-> Qwen2.5-VL-72B-Instruct
-> direct / contextual / irrelevant + Target Concept
-> direct entities -> SAM3
-> best_masked_crop.png
-> result.json

设计说明：
1. 只处理 ent_ids_1 / ent_ids_2 中实际存在图片的实体。
2. --use_name 只控制是否把实体名称加入 Qwen prompt，不包含 char feature。
3. Qwen 初次判断阶段只加载一次模型。
4. Qwen 完成后释放模型，再加载一次 SAM3 批量处理 direct 实体。
5. SAM3 无 mask 的实体最后统一重新加载 Qwen，二次确认 contextual / irrelevant。
6. 二次确认不会覆盖第一次生成的 target_concept。
7. 默认只保存 best_masked_crop.png；其他可视化结果由超参数控制。
8. result.json 支持断点续跑。
"""

import argparse
import gc
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Sam3Model, Sam3Processor
from qwen_vl_utils import process_vision_info

# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser(description="Batch entity-aware local segmentation for DBP15K")
parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--mllm_path", default="/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct", type=str)
parser.add_argument("--sam3_path", default="/mnt/DATA/chenxiaoli/MLLM/SAM3", type=str)
parser.add_argument("--output_dir", default="./data/DBP15K", type=str)
parser.add_argument("--semantic_info_name", default="ent_semantic_info.json", type=str)
parser.add_argument("--sam_threshold", default=0.5, type=float)
parser.add_argument("--mask_threshold", default=0.5, type=float)
parser.add_argument("--use_name", action="store_true", default=False)
parser.add_argument("--qwen_batch_size", default=8, type=int)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--save_every", default=10, type=int)
parser.add_argument("--max_entities", default=0, type=int)
parser.add_argument("--save_original", action="store_true", default=False)
parser.add_argument("--save_overlay", action="store_true", default=False)
parser.add_argument("--save_mask", action="store_true", default=False)
parser.add_argument("--save_masked", action="store_true", default=False)
parser.add_argument("--save_crop", action="store_true", default=False)
parser.add_argument("--save_raw_response", action="store_true", default=False)
args = parser.parse_args()

# ==================== Paths / Constants ====================
DATA_DIR = os.path.join(args.output_dir, args.data_split)
IMAGE_DIR = os.path.join(DATA_DIR, "concat_images")
NAME_DICT_PATH = os.path.join(DATA_DIR, "candidates", "name_dict")
SEMANTIC_INFO_PATH = os.path.join(DATA_DIR, args.semantic_info_name)

RUN_NAME = "with_name" if args.use_name else "without_name"
LOCAL_IMAGE_DIR = os.path.join(DATA_DIR, "seg_images", RUN_NAME)
OUTPUT_ROOT = LOCAL_IMAGE_DIR
DEBUG_DIR = os.path.join(OUTPUT_ROOT, "debug")
RESULT_PATH = os.path.join(OUTPUT_ROOT, "results.json")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".gif")
SYSTEM_PROMPT = "You are a helpful assistant."

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
if args.save_original or args.save_overlay or args.save_mask or args.save_masked or args.save_crop:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ==================== Utility ====================
def atomic_save_json(data, path):
    """原子写入 JSON，尽量避免运行中断导致 result.json 损坏。"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def load_existing_results():
    if args.overwrite or not os.path.exists(RESULT_PATH):
        return {}
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def read_entity_ids(path):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(int(line.split("\t")[0]))
    return ids

def build_image_map(valid_entity_ids):
    """扫描 concat_images，只保留 ent_ids_1 / ent_ids_2 中确实有图片的实体。"""
    image_map = {}
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    for file_name in os.listdir(IMAGE_DIR):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() not in VALID_EXTS or not stem.isdigit():
            continue
        entity_id = int(stem)
        if entity_id in valid_entity_ids:
            image_map[entity_id] = os.path.join(IMAGE_DIR, file_name)
    return image_map

def safe_open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")

def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def qwen_input_device(model):
    """device_map='auto' 时，把输入放到模型首个参数所在设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_progress(results):
    atomic_save_json(results, RESULT_PATH)

# ==================== Data Loading ====================
def load_name_dict():
    if not args.use_name:
        return {}
    if not os.path.exists(NAME_DICT_PATH):
        raise FileNotFoundError(f"Name dict not found: {NAME_DICT_PATH}")
    with open(NAME_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["ent"]

def load_semantic_info():
    if not os.path.exists(SEMANTIC_INFO_PATH):
        raise FileNotFoundError(f"Semantic info not found: {SEMANTIC_INFO_PATH}")
    with open(SEMANTIC_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ==================== Prompt ====================
def append_entity_context(prompt, entity_name, semantic):
    if args.use_name:
        prompt += f"The entity name is: {entity_name}.\n"

    prompt += "Entity attributes:\n"
    attributes = semantic.get("attributes", [])
    if attributes:
        for item in attributes:
            attribute = item.get("attribute", "")
            value = item.get("value", "")
            prompt += f"- {attribute}: {value}\n" if value else f"- {attribute}\n"
    else:
        prompt += "- none\n"

    prompt += "Entity relations:\n"
    relations = semantic.get("relations", [])
    if relations:
        for item in relations:
            direction = item.get("direction", "")
            relation = item.get("relation", "")
            neighbor = item.get("neighbor", "")
            if direction == "outgoing":
                prompt += f"- this entity --{relation}--> {neighbor}\n"
            elif direction == "incoming":
                prompt += f"- {neighbor} --{relation}--> this entity\n"
            else:
                prompt += f"- {relation}: {neighbor}\n"
    else:
        prompt += "- none\n"

    return prompt

def build_reasoning_prompt(entity_name, semantic):
    prompt = append_entity_context("", entity_name, semantic)
    prompt += (
        "Inspect the provided image and determine how the image visually represents this entity.\n"
        "Choose exactly one representation type:\n"
        "- direct: the image directly depicts the entity itself. Official logos, emblems, posters, covers, "
        "title cards, or other canonical visual identities of the entity also count as direct.\n"
        "- contextual: the image does not directly depict the entity itself, but depicts a visual object or scene strongly associated with it.\n"
        "- irrelevant: the image does not provide meaningful visual evidence for the entity.\n"
        "Then give one short visual target concept suitable for text-prompted segmentation. "
        "The target concept must be a concise English visual noun phrase describing the visible object or region "
        "that best represents or is visually associated with the entity in the image. "
        "Do not simply copy or translate the entity name as the target concept. "
        "Always output a target concept, regardless of whether the representation type is direct, contextual, or irrelevant.\n"
        "Output strictly in the following format:\n"
        "[REPRESENTATION TYPE] = direct/contextual/irrelevant\n"
        "[TARGET CONCEPT] = concise English visual noun phrase"
    )
    return prompt

def build_reconfirm_prompt(entity_name, semantic, failed_target_concept):
    prompt = append_entity_context("", entity_name, semantic)
    prompt += (
        f"The image was previously classified as direct with target concept '{failed_target_concept}', "
        "but SAM3 could not obtain any valid segmentation mask for that concept. "
        "Reconsider whether the image is contextual or irrelevant. "
        "Do not generate a new target concept; the previous target concept will be preserved.\n"
        "Choose exactly one:\n"
        "- contextual: the image does not directly depict the entity itself, but depicts a visual object or scene strongly associated with it.\n"
        "- irrelevant: the image does not provide meaningful visual evidence for the entity.\n"
        "Output strictly in the following format:\n"
        "[REPRESENTATION TYPE] = contextual/irrelevant"
    )
    return prompt

# ==================== Qwen ====================
def load_qwen():
    print("\nLoading Qwen2.5-VL-72B-Instruct ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.mllm_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor

def qwen_generate_batch(model, processor, images, prompts, max_new_tokens):
    conversations = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}] for image, prompt in zip(images, prompts)]
    texts, image_inputs, video_inputs = [], [], []
    for messages in conversations:
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        batch_images, batch_videos = process_vision_info(messages)
        if batch_images:
            image_inputs.extend(batch_images)
        if batch_videos:
            video_inputs.extend(batch_videos)
    inputs = processor(text=texts, images=image_inputs or None, videos=video_inputs or None, padding=True, return_tensors="pt").to(qwen_input_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del inputs, generated_ids
    return responses

def qwen_generate(model, processor, image, prompt, max_new_tokens):
    return qwen_generate_batch(model, processor, [image], [prompt], max_new_tokens)[0]

def parse_initial_response(response):
    type_match = re.search(r"\[REPRESENTATION TYPE\]\s*=\s*(direct|contextual|irrelevant)", response, re.I)
    concept_match = re.search(r"\[TARGET CONCEPT\]\s*=\s*([^\r\n]*)", response, re.I)
    representation_type = type_match.group(1).lower() if type_match else "unknown"
    target_concept = concept_match.group(1).strip().strip(".") if concept_match else ""
    if representation_type == "unknown" or not target_concept:
        return "unknown", target_concept
    return representation_type, target_concept

def parse_reconfirm_response(response):
    """先解析固定格式；若格式稍有偏差，再做一次宽松匹配。"""
    strict = re.search(r"\[REPRESENTATION TYPE\]\s*=\s*(contextual|irrelevant)", response, re.I)
    if strict:
        return strict.group(1).lower()

    contextual = re.search(r"\bcontextual\b", response, re.I)
    irrelevant = re.search(r"\birrelevant\b", response, re.I)
    if contextual and not irrelevant:
        return "contextual"
    if irrelevant and not contextual:
        return "irrelevant"
    return "unknown"

# ==================== SAM3 ====================
def load_sam3():
    print("\nLoading SAM3 ...")
    sam_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Sam3Model.from_pretrained(args.sam3_path).to(sam_device)
    processor = Sam3Processor.from_pretrained(args.sam3_path)
    model.eval()
    return model, processor, sam_device

def run_sam3(model, processor, sam_device, image, target_concept):
    inputs = processor(images=image, text=target_concept, return_tensors="pt").to(sam_device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(outputs, threshold=args.sam_threshold, mask_threshold=args.mask_threshold, target_sizes=inputs.get("original_sizes").tolist())[0]

    del inputs, outputs
    return results

def save_sam_result(entity_id, image, results):
    masks = results["masks"].detach().cpu()
    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()

    if len(masks) == 0:
        return {
            "sam_success": False,
            "sam_candidates": 0,
            "best_mask_score": None,
            "local_image": None,
        }

    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[best_idx].numpy().astype(bool)
    best_box = boxes[best_idx].numpy()
    best_score = float(scores[best_idx].item())

    x1, y1, x2, y2 = best_box
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(image.width, int(np.ceil(x2)))
    y2 = min(image.height, int(np.ceil(y2)))

    if x2 <= x1 or y2 <= y1 or not best_mask.any():
        return {
            "sam_success": False,
            "sam_candidates": int(len(masks)),
            "best_mask_score": best_score,
            "local_image": None,
        }

    image_np = np.array(image)
    masked_np = image_np.copy()
    masked_np[~best_mask] = 0
    masked_image = Image.fromarray(masked_np)

    # 默认唯一保存的局部图：best_masked_crop.png
    local_image_path = os.path.join(LOCAL_IMAGE_DIR, f"{entity_id}.jpg")
    masked_crop = masked_image.crop((x1, y1, x2, y2))
    masked_crop.convert("RGB").save(local_image_path, format="JPEG", quality=95)

    # 其他可视化输出默认关闭，仅用于 debug。
    if args.save_original or args.save_overlay or args.save_mask or args.save_masked or args.save_crop:
        entity_debug_dir = os.path.join(DEBUG_DIR, str(entity_id))
        os.makedirs(entity_debug_dir, exist_ok=True)

        if args.save_original:
            image.save(os.path.join(entity_debug_dir, "original.jpg"))

        if args.save_mask:
            mask_img = Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L")
            mask_img.save(os.path.join(entity_debug_dir, "best_mask.png"))

        if args.save_masked:
            masked_image.save(os.path.join(entity_debug_dir, "best_masked.png"))

        if args.save_crop:
            crop = image.crop((x1, y1, x2, y2))
            crop.save(os.path.join(entity_debug_dir, "best_crop.jpg"))

        if args.save_overlay:
            overlay = image.convert("RGBA")
            overlay_array = np.array(overlay)
            mask_layer = np.zeros_like(overlay_array)
            mask_layer[best_mask] = [255, 0, 0, 110]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_layer, mode="RGBA"))
            draw = ImageDraw.Draw(overlay)
            for i, (box, score) in enumerate(zip(boxes.numpy(), scores.numpy())):
                bx1, by1, bx2, by2 = box.tolist()
                draw.rectangle([bx1, by1, bx2, by2], outline="yellow", width=3)
                draw.text((bx1, by1), f"{i}: {score:.3f}", fill="yellow")
            overlay.save(os.path.join(entity_debug_dir, "sam3_overlay.png"))

    return {
        "sam_success": True,
        "sam_candidates": int(len(masks)),
        "best_mask_score": best_score,
        "local_image": local_image_path,
    }

# ==================== Statistics ====================
def pct(n, d):
    return 100.0 * n / d if d else 0.0

def collect_statistics(results, valid_entity_count, image_entity_count, elapsed):
    rows = [v for v in results.values() if isinstance(v, dict)]

    initial_counter = Counter(x.get("initial_representation_type", "not_processed") for x in rows)
    final_counter = Counter(x.get("final_representation_type", "not_processed") for x in rows)

    qwen_done = sum("initial_representation_type" in x for x in rows)
    sam_attempted = sum(bool(x.get("sam_attempted", False)) for x in rows)
    sam_success = sum(bool(x.get("sam_success", False)) for x in rows)
    sam_failed = sam_attempted - sam_success
    local_ready = sum(bool(x.get("local_image")) and os.path.exists(x.get("local_image", "")) for x in rows)
    reconfirmed = sum(bool(x.get("reconfirmed", False)) for x in rows)

    candidate_values = [x.get("sam_candidates") for x in rows if x.get("sam_success") and x.get("sam_candidates") is not None]
    score_values = [x.get("best_mask_score") for x in rows if x.get("sam_success") and x.get("best_mask_score") is not None]

    return {
        "dataset_entities": valid_entity_count,
        "entities_with_image": image_entity_count,
        "image_coverage": pct(image_entity_count, valid_entity_count),
        "qwen_processed": qwen_done,
        "initial_direct": initial_counter["direct"],
        "initial_contextual": initial_counter["contextual"],
        "initial_irrelevant": initial_counter["irrelevant"],
        "initial_unknown": initial_counter["unknown"],
        "sam_attempted": sam_attempted,
        "sam_success": sam_success,
        "sam_failed": sam_failed,
        "sam_success_rate": pct(sam_success, sam_attempted),
        "reconfirmed": reconfirmed,
        "final_direct": final_counter["direct"],
        "final_contextual": final_counter["contextual"],
        "final_irrelevant": final_counter["irrelevant"],
        "final_unknown": final_counter["unknown"],
        "local_ready": local_ready,
        "local_ready_rate_among_images": pct(local_ready, image_entity_count),
        "avg_sam_candidates_success": float(np.mean(candidate_values)) if candidate_values else 0.0,
        "avg_best_mask_score": float(np.mean(score_values)) if score_values else 0.0,
        "elapsed_seconds": elapsed,
    }

def print_statistics(stats):
    print("\n" + "=" * 72)
    print("DBP15K LOCAL SEGMENTATION STATISTICS")
    print("=" * 72)
    print(f"Data split                         : {args.data_split}")
    print(f"Use entity name                   : {args.use_name}")
    print(f"SAM threshold                     : {args.sam_threshold}")
    print(f"Mask threshold                    : {args.mask_threshold}")
    print("-" * 72)
    print(f"Dataset entities                  : {stats['dataset_entities']}")
    print(f"Entities with image               : {stats['entities_with_image']} ({stats['image_coverage']:.2f}%)")
    print(f"Qwen initial processed            : {stats['qwen_processed']}")
    print("-" * 72)
    print(f"Initial direct                    : {stats['initial_direct']} ({pct(stats['initial_direct'], stats['qwen_processed']):.2f}%)")
    print(f"Initial contextual                : {stats['initial_contextual']} ({pct(stats['initial_contextual'], stats['qwen_processed']):.2f}%)")
    print(f"Initial irrelevant                : {stats['initial_irrelevant']} ({pct(stats['initial_irrelevant'], stats['qwen_processed']):.2f}%)")
    print(f"Initial unknown / parse failure   : {stats['initial_unknown']} ({pct(stats['initial_unknown'], stats['qwen_processed']):.2f}%)")
    print("-" * 72)
    print(f"SAM3 attempted                    : {stats['sam_attempted']}")
    print(f"SAM3 success                      : {stats['sam_success']}")
    print(f"SAM3 no valid mask                : {stats['sam_failed']}")
    print(f"SAM3 success rate                 : {stats['sam_success_rate']:.2f}%")
    print(f"Reconfirmed by Qwen               : {stats['reconfirmed']}")
    print(f"Average SAM candidates (success)  : {stats['avg_sam_candidates_success']:.4f}")
    print(f"Average best mask score           : {stats['avg_best_mask_score']:.4f}")
    print("-" * 72)
    final_total = stats["final_direct"] + stats["final_contextual"] + stats["final_irrelevant"] + stats["final_unknown"]
    print(f"Final direct                      : {stats['final_direct']} ({pct(stats['final_direct'], final_total):.2f}%)")
    print(f"Final contextual                  : {stats['final_contextual']} ({pct(stats['final_contextual'], final_total):.2f}%)")
    print(f"Final irrelevant                  : {stats['final_irrelevant']} ({pct(stats['final_irrelevant'], final_total):.2f}%)")
    print(f"Final unknown                     : {stats['final_unknown']} ({pct(stats['final_unknown'], final_total):.2f}%)")
    print(f"Local images ready for CLIP       : {stats['local_ready']} ({stats['local_ready_rate_among_images']:.2f}% of image entities)")
    print("-" * 72)
    print(f"Total elapsed time                : {stats['elapsed_seconds'] / 3600.0:.3f} h")
    print(f"Result JSON                       : {RESULT_PATH}")
    print(f"Local image directory             : {LOCAL_IMAGE_DIR}")
    print("=" * 72)

# ==================== Main ====================
def main():
    start_time = time.time()

    ent_ids_1 = read_entity_ids(os.path.join(DATA_DIR, "ent_ids_1"))
    ent_ids_2 = read_entity_ids(os.path.join(DATA_DIR, "ent_ids_2"))
    valid_entity_ids = ent_ids_1 | ent_ids_2

    image_map = build_image_map(valid_entity_ids)
    entity_ids = sorted(image_map.keys())

    if args.max_entities > 0:
        entity_ids = entity_ids[:args.max_entities]
        image_map = {entity_id: image_map[entity_id] for entity_id in entity_ids}

    name_dict = load_name_dict()
    semantic_info = load_semantic_info()
    results = load_existing_results()

    print("=" * 72)
    print("DBP15K BATCH LOCAL SEGMENTATION")
    print("=" * 72)
    print(f"Data split            : {args.data_split}")
    print(f"Use entity name       : {args.use_name}")
    print(f"Qwen batch size       : {args.qwen_batch_size}")
    print(f"Dataset entities      : {len(valid_entity_ids)}")
    print(f"Entities with image   : {len(entity_ids)}")
    print(f"Resume result         : {RESULT_PATH}")
    print(f"Overwrite             : {args.overwrite}")
    print(f"Segmentation images   : {LOCAL_IMAGE_DIR}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Phase 1: Qwen 初次判断。Qwen 整个阶段只加载一次。
    # ------------------------------------------------------------------
    pending_initial = []
    for entity_id in entity_ids:
        record = results.get(str(entity_id), {})
        if args.overwrite or "initial_representation_type" not in record or "target_concept" not in record:
            pending_initial.append(entity_id)

    if pending_initial:
        qwen_model, qwen_processor = load_qwen()
        phase_start = time.time()

        progress = tqdm(total=len(pending_initial), desc="Qwen initial reasoning")
        for batch_start in range(0, len(pending_initial), max(1, args.qwen_batch_size)):
            batch_ids = pending_initial[batch_start:batch_start + max(1, args.qwen_batch_size)]
            valid_ids, images, prompts = [], [], []
            for entity_id in batch_ids:
                key = str(entity_id)
                record = results.get(key, {})
                record["entity_id"] = entity_id
                record["image_path"] = image_map[entity_id]
                record["use_name"] = args.use_name
                results[key] = record
                try:
                    image = safe_open_image(image_map[entity_id])
                    entity_name = name_dict.get(key, "") if args.use_name else ""
                    semantic = semantic_info.get(key, {"attributes": [], "relations": []})
                    valid_ids.append(entity_id)
                    images.append(image)
                    prompts.append(build_reasoning_prompt(entity_name, semantic))
                except Exception as e:
                    record["initial_representation_type"] = "unknown"
                    record["target_concept"] = record.get("target_concept", "")
                    record["final_representation_type"] = "unknown"
                    record["error"] = f"image_open_error: {type(e).__name__}: {e}"
            if valid_ids:
                try:
                    responses = qwen_generate_batch(qwen_model, qwen_processor, images, prompts, max_new_tokens=128)
                except Exception as batch_error:
                    release_cuda()
                    responses = []
                    for image, prompt in zip(images, prompts):
                        try:
                            responses.append(qwen_generate(qwen_model, qwen_processor, image, prompt, max_new_tokens=128))
                        except Exception as e:
                            responses.append(e)
                for entity_id, response in zip(valid_ids, responses):
                    key = str(entity_id)
                    record = results[key]
                    if isinstance(response, Exception):
                        record["initial_representation_type"] = "unknown"
                        record["target_concept"] = record.get("target_concept", "")
                        record["final_representation_type"] = "unknown"
                        record["error"] = f"qwen_initial_error: {type(response).__name__}: {response}"
                    else:
                        representation_type, target_concept = parse_initial_response(response)
                        record["initial_representation_type"] = representation_type
                        record["target_concept"] = target_concept
                        record["final_representation_type"] = representation_type
                        record["sam_attempted"] = False
                        record["sam_success"] = False
                        record["sam_candidates"] = None
                        record["best_mask_score"] = None
                        record["local_image"] = None
                        record["reconfirmed"] = False
                        record["error"] = None if representation_type != "unknown" and target_concept else "qwen_initial_parse_failed"
                        if args.save_raw_response:
                            record["initial_qwen_response"] = response
            progress.update(len(batch_ids))
            if progress.n % max(1, args.save_every) == 0 or progress.n == len(pending_initial):
                save_progress(results)
        progress.close()

        save_progress(results)
        print(f"Qwen initial phase time: {(time.time() - phase_start) / 3600.0:.3f} h")

        del qwen_model, qwen_processor
        release_cuda()
    else:
        print("\nQwen initial phase: nothing pending, skipped.")

    # ------------------------------------------------------------------
    # Phase 2: SAM3。只处理 initial direct + 非空 concept。
    # ------------------------------------------------------------------
    pending_sam = []
    for entity_id in entity_ids:
        record = results.get(str(entity_id), {})
        if record.get("initial_representation_type") != "direct":
            continue
        if not record.get("target_concept"):
            continue
        local_path = record.get("local_image")
        already_success = bool(record.get("sam_success")) and bool(local_path) and os.path.exists(local_path)
        already_failed = record.get("sam_attempted") is True and record.get("sam_success") is False
        if args.overwrite or (not already_success and not already_failed):
            pending_sam.append(entity_id)

    if pending_sam:
        sam_model, sam_processor, sam_device = load_sam3()
        phase_start = time.time()

        for idx, entity_id in enumerate(tqdm(pending_sam, desc="SAM3 segmentation"), 1):
            key = str(entity_id)
            record = results[key]
            image_path = image_map[entity_id]

            try:
                image = safe_open_image(image_path)
                sam_results = run_sam3(sam_model, sam_processor, sam_device, image, record["target_concept"])
                sam_info = save_sam_result(entity_id, image, sam_results)
                record.update(sam_info)
                record["sam_attempted"] = True
                record["final_representation_type"] = "direct" if record["sam_success"] else record["initial_representation_type"]
                record["error"] = None if record["sam_success"] else "sam_no_valid_mask"
            except Exception as e:
                record["sam_attempted"] = True
                record["sam_success"] = False
                record["sam_candidates"] = None
                record["best_mask_score"] = None
                record["local_image"] = None
                record["error"] = f"sam_error: {type(e).__name__}: {e}"

            results[key] = record

            if idx % max(1, args.save_every) == 0:
                save_progress(results)

        save_progress(results)
        print(f"SAM3 phase time: {(time.time() - phase_start) / 3600.0:.3f} h")

        del sam_model, sam_processor
        release_cuda()
    else:
        print("\nSAM3 phase: nothing pending, skipped.")

    # ------------------------------------------------------------------
    # Phase 3: SAM3 无有效 mask 的 direct 实体重新交给 Qwen。
    # target_concept 始终保留第一次生成的值，不会被二次确认覆盖。
    # ------------------------------------------------------------------
    pending_reconfirm = []
    for entity_id in entity_ids:
        record = results.get(str(entity_id), {})
        if record.get("initial_representation_type") == "direct" and record.get("sam_attempted") and not record.get("sam_success"):
            if args.overwrite or not record.get("reconfirmed", False):
                pending_reconfirm.append(entity_id)

    if pending_reconfirm:
        qwen_model, qwen_processor = load_qwen()
        phase_start = time.time()

        progress = tqdm(total=len(pending_reconfirm), desc="Qwen reconfirmation")
        for batch_start in range(0, len(pending_reconfirm), max(1, args.qwen_batch_size)):
            batch_ids = pending_reconfirm[batch_start:batch_start + max(1, args.qwen_batch_size)]
            valid_ids, images, prompts = [], [], []
            for entity_id in batch_ids:
                key = str(entity_id)
                record = results[key]
                try:
                    image = safe_open_image(image_map[entity_id])
                    entity_name = name_dict.get(key, "") if args.use_name else ""
                    semantic = semantic_info.get(key, {"attributes": [], "relations": []})
                    valid_ids.append(entity_id)
                    images.append(image)
                    prompts.append(build_reconfirm_prompt(entity_name, semantic, record.get("target_concept", "")))
                except Exception as e:
                    record["reconfirmed"] = True
                    record["reconfirmed_representation_type"] = "unknown"
                    record["final_representation_type"] = "unknown"
                    record["error"] = f"image_open_error: {type(e).__name__}: {e}"
            if valid_ids:
                try:
                    responses = qwen_generate_batch(qwen_model, qwen_processor, images, prompts, max_new_tokens=64)
                except Exception:
                    release_cuda()
                    responses = []
                    for image, prompt in zip(images, prompts):
                        try:
                            responses.append(qwen_generate(qwen_model, qwen_processor, image, prompt, max_new_tokens=64))
                        except Exception as e:
                            responses.append(e)
                for entity_id, response in zip(valid_ids, responses):
                    key = str(entity_id)
                    record = results[key]
                    original_concept = record.get("target_concept", "")
                    if isinstance(response, Exception):
                        record["reconfirmed"] = True
                        record["reconfirmed_representation_type"] = "unknown"
                        record["final_representation_type"] = "unknown"
                        record["target_concept"] = original_concept
                        record["error"] = f"qwen_reconfirm_error: {type(response).__name__}: {response}"
                    else:
                        confirmed_type = parse_reconfirm_response(response)
                        record["reconfirmed"] = True
                        record["reconfirmed_representation_type"] = confirmed_type
                        record["final_representation_type"] = confirmed_type
                        record["target_concept"] = original_concept
                        record["error"] = "sam_no_valid_mask" if confirmed_type != "unknown" else "reconfirm_parse_failed_after_sam_no_mask"
                        if args.save_raw_response:
                            record["reconfirm_qwen_response"] = response
            progress.update(len(batch_ids))
            if progress.n % max(1, args.save_every) == 0 or progress.n == len(pending_reconfirm):
                save_progress(results)
        progress.close()

        save_progress(results)
        print(f"Qwen reconfirmation phase time: {(time.time() - phase_start) / 3600.0:.3f} h")

        del qwen_model, qwen_processor
        release_cuda()
    else:
        print("\nQwen reconfirmation phase: nothing pending, skipped.")

    # 最终确保 contextual / irrelevant / unknown 没有 local image，后续 CLIP 直接补 zero vector。
    for entity_id in entity_ids:
        key = str(entity_id)
        record = results.get(key, {})
        if record.get("final_representation_type") != "direct":
            record["local_image"] = None
        results[key] = record

    save_progress(results)

    elapsed = time.time() - start_time
    stats = collect_statistics(results, len(valid_entity_ids), len(entity_ids), elapsed)
    print_statistics(stats)

if __name__ == "__main__":
    main()
