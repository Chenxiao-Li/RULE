import json

input_file = "./DBP15K/DNC_0.5_zh_en.json"
output_file = "./DBP15K/DNC_0.5_zh_en_candidate_statistics.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

ground_rank_0 = []
in_top10_not_top1 = []
out_top10 = []

for entity_id, item in data.items():
    entity_id = int(entity_id)
    ref = item["ref"]
    rank = item["ground_rank"]
    candidates = item["candidates"]

    if rank == 0:
        ground_rank_0.append(entity_id)

    elif ref in candidates:
        in_top10_not_top1.append(entity_id)

    else:
        out_top10.append(entity_id)

total = len(data)

result = {
    "summary": {
        "total": total,
        "ground_rank_0": {
            "count": len(ground_rank_0),
            "ratio": len(ground_rank_0) / total
        },
        "in_top10_not_top1": {
            "count": len(in_top10_not_top1),
            "ratio": len(in_top10_not_top1) / total
        },
        "out_top10": {
            "count": len(out_top10),
            "ratio": len(out_top10) / total
        }
    },
    "ground_rank_0": ground_rank_0,
    "in_top10_not_top1": in_top10_not_top1,
    "out_top10": out_top10
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"Saved to {output_file}")