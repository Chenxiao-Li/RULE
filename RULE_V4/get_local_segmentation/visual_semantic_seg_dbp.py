"""
DBP15K Entity-Centric Visual Semantic Views

Reference input/model-loading conventions are adapted from the user's original
local_segmentation_dbp.py:
- data/DBP15K/{zh_en|ja_en|fr_en}/
- ent_ids_1 / ent_ids_2
- concat_images/<entity_id>.{jpg,jpeg,png,gif}
- candidates/name_dict
- ent_semantic_info.json
- Qwen2.5-VL model path
- SAM3 model path

Idea:
1. Qwen receives entity semantics (+ optional entity name) and the image.
2. Qwen predicts one entity-centric visual semantic relation:
       (relation, visual_target)
   relation describes WHICH VISUAL ASPECT of the entity is represented by the image;
   visual_target is the concrete visible object/region supporting that relation.
3. SAM3 grounds each visual_target.
4. If SAM3 cannot produce a valid mask, that view is discarded.
5. If grounding succeeds, save a natural crop around the grounded region.
6. Each valid view is stored as:
       (relation, visual_target, crop)
7. The global image is untouched. This script only constructs non-global views.
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
parser = argparse.ArgumentParser(description="Entity-centric visual semantic view construction for DBP15K")
parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"], help="DBP15K language pair to process. Options: zh_en, ja_en, fr_en. Default: zh_en.",)
parser.add_argument("--mllm_path", default="/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct", type=str, help="Local Qwen2.5-VL model path. Qwen predicts the entity-image visual semantic relation and the concrete visual target for SAM3.",)
parser.add_argument("--sam3_path", default="/mnt/DATA/chenxiaoli/MLLM/SAM3", type=str, help="Local SAM3 model path. SAM3 segments the concrete visual target predicted by Qwen.",)
parser.add_argument("--output_dir", default="./data/DBP15K", type=str, help="Root directory containing DBP15K. Data are read from <output_dir>/<data_split>/. Default: ./data/DBP15K.",)
parser.add_argument("--semantic_info_name", default="ent_semantic_info.json", type=str, help="Entity semantic-information JSON filename under the selected split. It should contain entity attributes and relations.",)
parser.add_argument("--sam_threshold", default=0.5, type=float, help="SAM3 object confidence threshold. Higher values keep only more confident segmentation candidates. Default: 0.5.",)
parser.add_argument("--mask_threshold", default=0.5, type=float, help="SAM3 pixel-level mask threshold. Higher values produce stricter binary masks. Default: 0.5.",)
parser.add_argument("--use_name", action="store_false", default=True, help="Include the entity name from candidates/name_dict in the Qwen prompt. Without this flag, only semantic information and image are used.",)
parser.add_argument("--qwen_batch_size", default=16, type=int, help="Number of images processed by Qwen per batch. Lower it if GPU memory is unstable. Default: 2.")
parser.add_argument("--qwen_max_pixels", default=262144, type=int, help="Maximum pixels for each image sent to Qwen after aspect-ratio-preserving resize. Lower it to reduce VRAM. Default: 262144 (about 512x512).")
parser.add_argument("--crop_expand", default=0.15, type=float, help="Expansion ratio around the SAM3 bounding box before cropping. 0.15 keeps about 15 percent extra surrounding context. Default: 0.15.",)
parser.add_argument("--min_box_ratio", default=0.002, type=float, help="Minimum SAM3 bounding-box area divided by whole-image area. Smaller detections are discarded. Default: 0.002.",)
parser.add_argument("--max_box_ratio", default=0.95, type=float, help="Maximum SAM3 bounding-box area divided by whole-image area. Larger detections are discarded as too broad. Default: 0.95.",)
parser.add_argument("--overwrite", action="store_true", default=False, help="Recompute entities even if previous results exist. Without this flag, completed entities are skipped.",)
parser.add_argument("--save_every", default=10, type=int, help="Save results.json after every N processed entities. Default: 10.",)
parser.add_argument("--max_entities", default=0, type=int, help="Maximum number of entities to process. 0 means all entities. Example: --max_entities 100 for debugging.",)
parser.add_argument("--save_original", action="store_true", default=False, help="Save the original image in the debug directory.",)
parser.add_argument("--save_overlay", action="store_true", default=False, help="Save a debug image showing the SAM3 mask and bounding box.",)
parser.add_argument("--save_mask", action="store_true", default=False, help="Save the binary SAM3 segmentation mask.",)
parser.add_argument("--save_raw_response", action="store_true", default=False, help="Save Qwen's raw response in results.json for inspecting relation and visual_target predictions.",)
args = parser.parse_args()

# ==================== Paths / Constants ====================
DATA_DIR = os.path.join(args.output_dir, args.data_split)
IMAGE_DIR = os.path.join(DATA_DIR, "concat_images")
NAME_DICT_PATH = os.path.join(DATA_DIR, "candidates", "name_dict")
SEMANTIC_INFO_PATH = os.path.join(DATA_DIR, args.semantic_info_name)

RUN_NAME = "with_name" if args.use_name else "without_name"
OUTPUT_ROOT = os.path.join(DATA_DIR, "seg_images", RUN_NAME)
SEG_IMAGE_ROOT = OUTPUT_ROOT
DEBUG_ROOT = os.path.join(OUTPUT_ROOT, "debug")
RESULT_PATH = os.path.join(OUTPUT_ROOT, "results.json")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".gif")
SYSTEM_PROMPT = "You are a careful visual semantic reasoning assistant for multimodal entity alignment."

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(SEG_IMAGE_ROOT, exist_ok=True)
if args.save_original or args.save_overlay or args.save_mask:
    os.makedirs(DEBUG_ROOT, exist_ok=True)

# ==================== Utility ====================
def atomic_save_json(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_existing_results():
    if args.overwrite or not os.path.exists(RESULT_PATH):
        return {}
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(results):
    atomic_save_json(results, RESULT_PATH)


def read_entity_ids(path):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line.split("\t")[0]))
    return ids


def build_image_map(valid_entity_ids):
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
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sanitize_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip().strip(".")


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def expand_box(box, image_width, image_height, expand_ratio):
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x1 -= width * expand_ratio
    y1 -= height * expand_ratio
    x2 += width * expand_ratio
    y2 += height * expand_ratio
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(image_width, int(np.ceil(x2))),
        min(image_height, int(np.ceil(y2))),
    )

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


def build_visual_semantic_prompt(entity_name, semantic):
    prompt = append_entity_context("", entity_name, semantic)
    prompt += f"""
Inspect the provided image together with the entity semantic information.

The original image relation is only a generic imageOf relation. Refine it into one or more entity-centric visual semantic relations describing WHICH VISUAL ASPECT of the entity is represented by this image.

Important rules:
1. The relation must describe how the visible content represents the entity itself.
2. Do NOT output scene-level relations between objects such as holding, standing_at, wearing, next_to, or in_front_of.
3. Do NOT use coarse labels such as direct, contextual, or irrelevant.
4. Good relation examples include: logo, portrait, product, headquarters, landmark, landscape, architecture, exterior, interior, storefront, poster, cover, jersey, stadium, team_photo, vehicle, monument, map, flag, emblem, signature, artwork, event_scene, natural_scenery, historical_site.
5. You may create another concise relation when none of the examples fits.
6. Different views should correspond to genuinely different visual aspects of the entity.
7. For every relation, provide one concrete visual_target that visibly supports that relation.
8. visual_target must be a concrete English visual noun phrase suitable for SAM3 text-prompted segmentation.
9. visual_target must be visibly present in the current image and must not be an abstract concept.
10. If no reliable entity-related visual aspect can be grounded, return an empty list.
11. Return exactly one view when a reliable entity-related visual aspect can be grounded.

Examples:
Company + logo image -> relation: logo, visual_target: company logo
University + tower image -> relation: landmark, visual_target: campus tower
National park + mountain image -> relation: landscape, visual_target: mountain landscape
Film + poster image -> relation: poster, visual_target: movie poster
Sports team + jersey image -> relation: jersey, visual_target: team jersey

Return STRICT JSON only:
{{
  "visual_views": [
    {{
      "relation": "concise entity-centric visual relation",
      "visual_target": "concrete English visual noun phrase"
    }}
  ]
}}
"""
    return prompt

def resize_for_qwen(image):
    max_pixels = max(1, int(args.qwen_max_pixels))
    width, height = image.size
    pixels = width * height

    if pixels <= max_pixels:
        return image

    scale = (max_pixels / float(pixels)) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# ==================== Qwen ====================
def load_qwen():
    print("\nLoading Qwen2.5-VL ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.mllm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def qwen_generate_batch(model, processor, images, prompts, max_new_tokens):
    conversations = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ] for image, prompt in zip(images, prompts)]

    texts, image_inputs, video_inputs = [], [], []
    for messages in conversations:
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        batch_images, batch_videos = process_vision_info(messages)
        if batch_images:
            image_inputs.extend(batch_images)
        if batch_videos:
            video_inputs.extend(batch_videos)

    inputs = processor(
        text=texts,
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(qwen_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del inputs, generated_ids
    return responses


def extract_json_object(response):
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.I)
    response = re.sub(r"\s*```$", "", response)
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in Qwen response")
    return json.loads(response[start:end + 1])


def parse_visual_semantic_response(response):
    data = extract_json_object(response)
    raw_views = data.get("visual_views", [])
    if not isinstance(raw_views, list):
        raise ValueError("visual_views is not a list")

    views, seen = [], set()
    for item in raw_views:
        if not isinstance(item, dict):
            continue
        relation = sanitize_text(item.get("relation", ""))
        visual_target = sanitize_text(item.get("visual_target", ""))
        if not relation or not visual_target:
            continue
        key = (relation.lower(), visual_target.lower())
        if key in seen:
            continue
        seen.add(key)
        views.append({"relation": relation, "visual_target": visual_target})
        break
    return views

# ==================== SAM3 ====================
def load_sam3():
    print("\nLoading SAM3 ...")
    sam_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Sam3Model.from_pretrained(args.sam3_path).to(sam_device)
    processor = Sam3Processor.from_pretrained(args.sam3_path)
    model.eval()
    return model, processor, sam_device


def run_sam3(model, processor, sam_device, image, visual_target):
    inputs = processor(images=image, text=visual_target, return_tensors="pt").to(sam_device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.sam_threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    del inputs, outputs
    return results


def select_best_valid_candidate(results, image):
    masks = results["masks"].detach().cpu()
    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()

    candidate_count = int(len(masks))
    if candidate_count == 0:
        return None

    order = torch.argsort(scores, descending=True).tolist()
    image_area = float(image.width * image.height)

    for idx in order:
        mask = masks[idx].numpy().astype(bool)
        box = boxes[idx].numpy()
        score = float(scores[idx].item())
        if not mask.any():
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        box_ratio = box_area / image_area if image_area > 0 else 0.0
        if box_ratio < args.min_box_ratio or box_ratio > args.max_box_ratio:
            continue

        return {
            "mask": mask,
            "box": box,
            "score": score,
            "candidate_count": candidate_count,
            "box_ratio": float(box_ratio),
        }

    return None


def save_grounded_view(entity_id, view_index, image, relation, visual_target, candidate):
    entity_view_dir = os.path.join(SEG_IMAGE_ROOT, str(entity_id))
    os.makedirs(entity_view_dir, exist_ok=True)

    crop_box = expand_box(candidate["box"], image.width, image.height, args.crop_expand)
    x1, y1, x2, y2 = crop_box
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop(crop_box).convert("RGB")
    view_path = os.path.join(entity_view_dir, f"view_{view_index}.jpg")
    crop.save(view_path, format="JPEG", quality=95)

    if args.save_original or args.save_overlay or args.save_mask:
        debug_dir = os.path.join(DEBUG_ROOT, str(entity_id), f"view_{view_index}")
        os.makedirs(debug_dir, exist_ok=True)

        if args.save_original:
            image.save(os.path.join(debug_dir, "original.jpg"))

        if args.save_mask:
            mask_img = Image.fromarray((candidate["mask"].astype(np.uint8) * 255), mode="L")
            mask_img.save(os.path.join(debug_dir, "mask.png"))

        if args.save_overlay:
            overlay = image.convert("RGBA")
            overlay_array = np.array(overlay)
            mask_layer = np.zeros_like(overlay_array)
            mask_layer[candidate["mask"]] = [255, 0, 0, 110]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_layer, mode="RGBA"))
            draw = ImageDraw.Draw(overlay)
            bx1, by1, bx2, by2 = candidate["box"].tolist()
            draw.rectangle([bx1, by1, bx2, by2], outline="yellow", width=3)
            draw.text((bx1, by1), f"{relation} | {visual_target} | {candidate['score']:.3f}", fill="yellow")
            overlay.save(os.path.join(debug_dir, "overlay.png"))

    return {
        "relation": relation,
        "visual_target": visual_target,
        "sam_score": candidate["score"],
        "sam_candidates": candidate["candidate_count"],
        "box_ratio": candidate["box_ratio"],
        "crop_box": [int(v) for v in crop_box],
        "view_path": view_path,
    }

# ==================== Statistics ====================
def collect_statistics(results, valid_entity_count, image_entity_count, elapsed):
    rows = [v for v in results.values() if isinstance(v, dict)]
    qwen_done = sum(bool(x.get("qwen_done", False)) for x in rows)
    proposed_views = sum(len(x.get("proposed_visual_views", [])) for x in rows)
    grounded_views = sum(len(x.get("grounded_visual_views", [])) for x in rows)
    entities_with_grounded = sum(len(x.get("grounded_visual_views", [])) > 0 for x in rows)
    failed_views = sum(sum(1 for v in x.get("grounding_attempts", []) if not v.get("kept", False)) for x in rows)
    duplicate_views = sum(int(x.get("duplicate_views_dropped", 0)) for x in rows)

    relation_counter = Counter()
    for row in rows:
        for view in row.get("grounded_visual_views", []):
            relation = view.get("relation")
            if relation:
                relation_counter[relation] += 1

    return {
        "dataset_entities": valid_entity_count,
        "entities_with_image": image_entity_count,
        "qwen_processed": qwen_done,
        "proposed_views": proposed_views,
        "grounded_views": grounded_views,
        "failed_or_dropped_views": failed_views,
        "duplicate_views_dropped": duplicate_views,
        "entities_with_grounded_views": entities_with_grounded,
        "relation_counter": relation_counter,
        "elapsed_seconds": elapsed,
    }


def print_statistics(stats):
    print("\n" + "=" * 72)
    print("DBP15K ENTITY-CENTRIC VISUAL SEMANTIC VIEW STATISTICS")
    print("=" * 72)
    print(f"Data split                         : {args.data_split}")
    print(f"Use entity name                   : {args.use_name}")
    print(f"SAM threshold                     : {args.sam_threshold}")
    print(f"Mask threshold                    : {args.mask_threshold}")
    print(f"Crop expansion                    : {args.crop_expand}")
    print("-" * 72)
    print(f"Dataset entities                  : {stats['dataset_entities']}")
    print(f"Entities with image               : {stats['entities_with_image']}")
    print(f"Qwen processed                    : {stats['qwen_processed']}")
    print(f"Proposed visual views             : {stats['proposed_views']}")
    print(f"Grounded visual views             : {stats['grounded_views']}")
    print(f"Failed / dropped views            : {stats['failed_or_dropped_views']}")
    print(f"Duplicate views dropped           : {stats['duplicate_views_dropped']}")
    print(f"Entities with grounded views      : {stats['entities_with_grounded_views']}")
    print("-" * 72)
    print("Top grounded visual relations:")
    for relation, count in stats["relation_counter"].most_common(20):
        print(f"  {relation:<32}: {count}")
    print("-" * 72)
    print(f"Elapsed                           : {stats['elapsed_seconds'] / 3600.0:.3f} h")
    print(f"Result JSON                       : {RESULT_PATH}")
    print(f"Seg image directory                    : {SEG_IMAGE_ROOT}")
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
    print("DBP15K ENTITY-CENTRIC VISUAL SEMANTIC VIEW CONSTRUCTION")
    print("=" * 72)
    print(f"Data split            : {args.data_split}")
    print(f"Use entity name       : {args.use_name}")
    print(f"Qwen batch size       : {args.qwen_batch_size}")
    print(f"Qwen max pixels       : {args.qwen_max_pixels}")
    print(f"Dataset entities      : {len(valid_entity_ids)}")
    print(f"Entities with image   : {len(entity_ids)}")
    print(f"Result JSON           : {RESULT_PATH}")
    print(f"Seg image directory        : {SEG_IMAGE_ROOT}")
    print("=" * 72)

    # Phase 1: Qwen predicts entity-centric visual semantic relations + visual targets.
    pending_qwen = []
    for entity_id in entity_ids:
        record = results.get(str(entity_id), {})
        if args.overwrite or not record.get("qwen_done", False):
            pending_qwen.append(entity_id)

    if pending_qwen:
        qwen_model, qwen_processor = load_qwen()
        progress = tqdm(total=len(pending_qwen), desc="Qwen visual semantic reasoning")

        for batch_start in range(0, len(pending_qwen), max(1, args.qwen_batch_size)):
            batch_ids = pending_qwen[batch_start:batch_start + max(1, args.qwen_batch_size)]
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
                    images.append(resize_for_qwen(image))
                    prompts.append(build_visual_semantic_prompt(entity_name, semantic))
                except Exception as e:
                    record["qwen_done"] = True
                    record["proposed_visual_views"] = []
                    record["qwen_error"] = f"image_open_error: {type(e).__name__}: {e}"

            if valid_ids:
                try:
                    responses = qwen_generate_batch(qwen_model, qwen_processor, images, prompts, max_new_tokens=320)
                except Exception:
                    release_cuda()
                    responses = []
                    for image, prompt in zip(images, prompts):
                        try:
                            responses.append(qwen_generate_batch(qwen_model, qwen_processor, [image], [prompt], max_new_tokens=320)[0])
                        except Exception as e:
                            responses.append(e)

                for entity_id, response in zip(valid_ids, responses):
                    key = str(entity_id)
                    record = results[key]
                    record["qwen_done"] = True

                    if isinstance(response, Exception):
                        record["proposed_visual_views"] = []
                        record["qwen_error"] = f"qwen_error: {type(response).__name__}: {response}"
                    else:
                        try:
                            record["proposed_visual_views"] = parse_visual_semantic_response(response)
                            record["qwen_error"] = None
                        except Exception as e:
                            record["proposed_visual_views"] = []
                            record["qwen_error"] = f"qwen_parse_error: {type(e).__name__}: {e}"

                        if args.save_raw_response:
                            record["qwen_response"] = response

                    record["grounding_done"] = False
                    record["grounding_attempts"] = []
                    record["grounded_visual_views"] = []
                    record["duplicate_views_dropped"] = 0

            progress.update(len(batch_ids))
            if progress.n % max(1, args.save_every) == 0 or progress.n == len(pending_qwen):
                save_progress(results)

        progress.close()
        save_progress(results)
        del qwen_model, qwen_processor
        release_cuda()
    else:
        print("\nQwen phase: nothing pending, skipped.")

    # Phase 2: SAM3 grounds every proposed visual_target.
    pending_sam = []
    for entity_id in entity_ids:
        key = str(entity_id)
        record = results.get(key, {})
        if not record.get("qwen_done", False):
            continue

        proposed_views = record.get("proposed_visual_views", [])
        if not proposed_views:
            record["grounding_done"] = True
            record["grounding_attempts"] = []
            record["grounded_visual_views"] = []
            results[key] = record
            continue

        if args.overwrite or not record.get("grounding_done", False):
            pending_sam.append(entity_id)

    if pending_sam:
        sam_model, sam_processor, sam_device = load_sam3()

        for idx, entity_id in enumerate(tqdm(pending_sam, desc="SAM3 visual grounding"), 1):
            key = str(entity_id)
            record = results[key]
            attempts = []
            grounded_views = []
            accepted_masks = []
            duplicate_count = 0

            try:
                image = safe_open_image(image_map[entity_id])

                for proposed in record.get("proposed_visual_views", [])[:args.max_views]:
                    relation = proposed["relation"]
                    visual_target = proposed["visual_target"]
                    attempt = {
                        "relation": relation,
                        "visual_target": visual_target,
                        "sam_success": False,
                        "kept": False,
                    }

                    try:
                        sam_results = run_sam3(sam_model, sam_processor, sam_device, image, visual_target)
                        candidate = select_best_valid_candidate(sam_results, image)

                        if candidate is None:
                            attempt["error"] = "sam_no_valid_candidate"
                            attempts.append(attempt)
                            continue

                        duplicate_of = None
                        max_iou = 0.0
                        for kept_idx, kept_mask in enumerate(accepted_masks):
                            iou = mask_iou(candidate["mask"], kept_mask)
                            max_iou = max(max_iou, iou)
                            if False:
                                duplicate_of = kept_idx
                                break

                        if duplicate_of is not None:
                            attempt.update({
                                "sam_success": True,
                                "sam_score": candidate["score"],
                                "drop_reason": "duplicate_region",
                                "duplicate_of": duplicate_of,
                                "max_iou": max_iou,
                            })
                            duplicate_count += 1
                            attempts.append(attempt)
                            continue

                        view_index = len(grounded_views)
                        saved_view = save_grounded_view(
                            entity_id,
                            view_index,
                            image,
                            relation,
                            visual_target,
                            candidate,
                        )

                        if saved_view is None:
                            attempt["error"] = "invalid_crop"
                            attempts.append(attempt)
                            continue

                        accepted_masks.append(candidate["mask"])
                        grounded_views.append(saved_view)
                        attempt.update({
                            "sam_success": True,
                            "kept": True,
                            "sam_score": candidate["score"],
                            "view_path": saved_view["view_path"],
                        })
                        attempts.append(attempt)

                    except Exception as e:
                        attempt["error"] = f"sam_error: {type(e).__name__}: {e}"
                        attempts.append(attempt)

            except Exception as e:
                record["grounding_error"] = f"image_open_error: {type(e).__name__}: {e}"

            record["grounding_attempts"] = attempts
            record["grounded_visual_views"] = grounded_views
            record["duplicate_views_dropped"] = duplicate_count
            record["grounding_done"] = True
            results[key] = record

            if idx % max(1, args.save_every) == 0:
                save_progress(results)

        save_progress(results)
        del sam_model, sam_processor
        release_cuda()
    else:
        print("\nSAM3 phase: nothing pending, skipped.")

    save_progress(results)

    elapsed = time.time() - start_time
    stats = collect_statistics(results, len(valid_entity_ids), len(entity_ids), elapsed)
    print_statistics(stats)


if __name__ == "__main__":
    main()
