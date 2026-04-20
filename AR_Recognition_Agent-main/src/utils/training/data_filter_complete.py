import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoProcessor

from qwenvl.train.argument import DataArguments
from data_processor import LazySupervisedDataset


# =========================
# 1. 构建 dataset
# =========================

card = "/home/jyinap/shared_storage/jyinap/vllm/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(
    card, trust_remote_code=True
)

data_args = DataArguments()
data_args.dataset_use = "egoit_99k"
data_args.model_type = "qwen3vl"

dataset = LazySupervisedDataset(
    processor,
    data_args=data_args,
)

print(f"Dataset size: {len(dataset)}")


# =========================
# 2. 多线程检查函数
# =========================

bad_indices = []
lock = threading.Lock()


def check_one(i: int):
    try:
        _ = dataset[i]
        return None
    except Exception as e:
        with lock:
            bad_indices.append(i)
        return i, str(e)


# =========================
# 3. 多线程扫描
# =========================

num_workers = 40   # 👈 可按 CPU 调，大于 dataloader worker 数即可

futures = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for i in range(len(dataset)):
        futures.append(executor.submit(check_one, i))

    for f in tqdm(as_completed(futures), total=len(futures)):
        _ = f.result()


bad_indices = sorted(set(bad_indices))

print(f"\nFound {len(bad_indices)} bad samples")


# =========================
# 4. 清洗 list_data_dict
# =========================

old_data = dataset.list_data_dict
old_len = len(old_data)

bad_set = set(bad_indices)

new_data = [
    sample
    for idx, sample in enumerate(old_data)
    if idx not in bad_set
]

print(f"Cleaned dataset size: {len(new_data)} (removed {old_len - len(new_data)})")


# =========================
# 5. 保存为 json
# =========================

output_path = "/home/jyinap/shared_storage/jyinap/dataset/EgoIT-99K/datasets/egoit_99k_cleaned.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"Saved cleaned dataset to {output_path}")
