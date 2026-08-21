"""
DBP15K 单实体局部视觉分割 Demo:
Entity Name + Image -> Qwen2.5-VL-72B-Instruct -> Representation Type + Target Concept
-> SAM3 -> Mask / Crop / Masked Crop
"""
import argparse
import os
import re
import json
import gc

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Sam3Model, Sam3Processor
from qwen_vl_utils import process_vision_info


# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser(description="Test entity-aware local segmentation for DBP15K")
parser.add_argument("--data_split", default="zh_en", type=str, choices=["zh_en", "ja_en", "fr_en"])
parser.add_argument("--entity_id", default=0, type=int)
parser.add_argument("--sam3_path", default="/mnt/DATA/chenxiaoli/MLLM/SAM3", type=str)
parser.add_argument("--output_dir", default="./output/local_segmentation_dbp", type=str)
parser.add_argument("--sam_threshold", default=0.5, type=float)
parser.add_argument("--mask_threshold", default=0.5, type=float)
args = parser.parse_args()


# ==================== Derived Settings ====================
data_file_path = os.path.join("./data", "DBP15K", args.data_split)
name_dict_path = os.path.join(data_file_path, "candidates", "name_dict")
# 修改：根据实体 ID 自动匹配 jpg/jpeg/png/gif 格式，扩展名大小写均可
def find_entity_image():
    image_dir = os.path.join(data_file_path, "concat_images")
    valid_exts = (".jpg", ".jpeg", ".png", ".gif")
    for file_name in os.listdir(image_dir):
        name, ext = os.path.splitext(file_name)
        if name == str(args.entity_id) and ext.lower() in valid_exts:
            return os.path.join(image_dir, file_name)
    raise FileNotFoundError(f"Image not found for entity {args.entity_id} in {image_dir}")

image_path = find_entity_image()
output_dir = os.path.join(args.output_dir, args.data_split, str(args.entity_id))
os.makedirs(output_dir, exist_ok=True)

MLLM_PATH = "/mnt/DATA/chenxiaoli/MLLM/Qwen2.5-VL/Qwen2.5-VL-72B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


# ==================== Data Loading ====================
def load_entity_name():
    with open(name_dict_path, "r", encoding="utf-8") as f:
        name_dict = json.load(f)["ent"]
    return name_dict[str(args.entity_id)]


# ==================== Qwen2.5-VL Reasoning ====================
def build_reasoning_prompt(entity_name):
    # 新增：先仅使用实体名字 + 图片，让 Qwen 判断图片如何表示实体，并生成适合 SAM3 的短概念
    return (
        f"The entity name is: {entity_name}.\n"
        "Inspect the provided image and determine how the image visually represents this entity.\n"
        "Choose exactly one representation type:\n"
        "- direct: the image directly depicts the entity itself.\n"
        "- contextual: the image does not directly depict the entity itself, but depicts a visual object or scene strongly associated with it.\n"
        "- irrelevant: the image does not provide meaningful visual evidence for the entity.\n"
        "Then give one short visual target concept suitable for text-prompted segmentation. "
        # 修改：Target Concept 必须是适合 SAM3 的英文视觉概念，不直接输出实体专名
        "The target concept must be a concise English visual noun phrase describing the visible object or region "
        "that best represents or is visually associated with the entity in the image, such as person, man, woman, "
        "man in a suit, flag, building, logo, map, car, or airplane. "
        "Do not simply copy or translate the entity name as the target concept. "
        # 修改：即使是 contextual 或 irrelevant，也仍然要求输出 Target Concept，便于后续分析
        "Always output a target concept, regardless of whether the representation type is direct, contextual, or irrelevant.\n"
        "Output strictly in the following format:\n"
        "[REPRESENTATION TYPE] = direct/contextual/irrelevant\n"
        "[TARGET CONCEPT] = concise English visual noun phrase"
    )


def qwen_reason(entity_name, image):
    # 新增：调用方式保持与 TTR_DBP.py 一致
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MLLM_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MLLM_PATH)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_reasoning_prompt(entity_name)},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 新增：Qwen 推理完成后立即释放 72B 模型，避免和 SAM3 同时占显存
    del inputs, generated_ids, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return response


def parse_qwen_response(response):
    # 新增：解析固定格式的 Representation Type 和 Target Concept
    type_match = re.search(r"\[REPRESENTATION TYPE\]\s*=\s*(direct|contextual|irrelevant)", response, re.I)
    concept_match = re.search(r"\[TARGET CONCEPT\]\s*=\s*(.+)", response, re.I)

    representation_type = type_match.group(1).lower() if type_match else "unknown"
    target_concept = concept_match.group(1).strip() if concept_match else ""
    target_concept = target_concept.splitlines()[0].strip().strip(".")

    return representation_type, target_concept


# ==================== SAM3 Segmentation ====================
def run_sam3(image, target_concept):
    # 新增：按照 Hugging Face SAM3 的 text-only prompt 接口进行分割
    # 修改：SAM3 模型较小，固定放在单张 GPU 上，避免 device_map="auto" 跨卡导致 tensor device 不一致
    sam_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(args.sam3_path).to(sam_device)
    processor = Sam3Processor.from_pretrained(args.sam3_path)

    inputs = processor(images=image, text=target_concept, return_tensors="pt").to(sam_device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.sam_threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    del inputs, outputs, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ==================== Visualization ====================
def save_segmentation_results(image, results, target_concept):
    # 新增：保存原图、全部候选 mask 可视化，以及最高分 mask 的 crop / mask / masked crop
    image.save(os.path.join(output_dir, "original.jpg"))

    masks = results["masks"].detach().cpu()
    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()

    if len(masks) == 0:
        print("SAM3 found no mask.")
        return

    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[best_idx].numpy().astype(bool)
    best_box = boxes[best_idx].numpy()

    overlay = image.convert("RGBA")
    overlay_array = np.array(overlay)
    mask_layer = np.zeros_like(overlay_array)
    mask_layer[best_mask] = [255, 0, 0, 110]
    overlay = Image.alpha_composite(overlay, Image.fromarray(mask_layer, mode="RGBA"))

    draw = ImageDraw.Draw(overlay)
    for i, (box, score) in enumerate(zip(boxes.numpy(), scores.numpy())):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, y1), f"{i}: {score:.3f}", fill="yellow")

    overlay.save(os.path.join(output_dir, "sam3_overlay.png"))

    mask_img = Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(os.path.join(output_dir, "best_mask.png"))

    image_np = np.array(image)
    masked_np = image_np.copy()
    masked_np[~best_mask] = 0
    masked_image = Image.fromarray(masked_np)
    masked_image.save(os.path.join(output_dir, "best_masked.png"))

    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2)))

    crop = image.crop((x1, y1, x2, y2))
    crop.save(os.path.join(output_dir, "best_crop.jpg"))

    masked_crop = masked_image.crop((x1, y1, x2, y2))
    masked_crop.save(os.path.join(output_dir, "best_masked_crop.png"))

    print(f"SAM3 candidates: {len(masks)}")
    print(f"Best mask index: {best_idx}")
    print(f"Best mask score: {float(scores[best_idx]):.4f}")
    print(f"Target concept: {target_concept}")
    print(f"Results saved to: {output_dir}")


# ==================== Entry Point ====================
if __name__ == "__main__":
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(name_dict_path):
        raise FileNotFoundError(f"Name dict not found: {name_dict_path}")

    entity_name = load_entity_name()
    image = Image.open(image_path).convert("RGB")

    print(f"Entity ID: {args.entity_id}")
    print(f"Entity Name: {entity_name}")
    print(f"Image: {image_path}")

    response = qwen_reason(entity_name, image)
    print("\n===== Qwen2.5-VL Response =====")
    print(response)

    representation_type, target_concept = parse_qwen_response(response)
    print(f"\nRepresentation Type: {representation_type}")
    print(f"Target Concept: {target_concept}")

    # 修改：只有 direct 类型进入 SAM3；contextual 和 irrelevant 均只保留 Qwen 的 Target Concept
    if representation_type != "direct" or not target_concept:
        print("Skip SAM3 because only direct representations are segmented, or no valid target concept was generated.")
    else:
        results = run_sam3(image, target_concept)
        save_segmentation_results(image, results, target_concept)
