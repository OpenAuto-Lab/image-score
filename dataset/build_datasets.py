import random
import json
from datasets import load_dataset

# ds = load_dataset("ILSVRC/imagenet-1k", split="train")
ds = load_dataset("zh-plus/tiny-imagenet", split="train") # use tiny for test 
label_names = ds.features["label"].names

with open("dataset/data/imagenet_templates.json", "r") as f:
    templates = json.load(f)
with open("dataset/data/imagenet_class_index_formatted.json", "r") as f:
    class_index = json.load(f)

def get_prompts_batched(batch, num_prompts=10):
    new_images = []
    new_labels = []
    new_prompts = []
    new_prompt_ids = []

    for img, label in zip(batch["image"], batch["label"]):
        cur_img_prompts, cur_img_prompt_ids = [], []
        for i in range(num_prompts):
            label_class = class_index.get(label_names[label], None)
            if not label_class:
                continue
            prompt = random.choice(templates).format(label_class.replace("_", " "))
            cur_img_prompts.append(prompt)
            cur_img_prompt_ids.append(i)
        new_images.append(img)
        new_labels.append(label)
        new_prompts.append(cur_img_prompts)
        new_prompt_ids.append(cur_img_prompt_ids)
    return {
        "image": new_images,
        "label": new_labels,
        "prompt": new_prompts,
        "prompt_id": new_prompt_ids
    }


ds_expanded = ds.map(
    lambda batch: get_prompts_batched(batch, num_prompts=1),
    batched=True,
    batch_size=16
)

print(ds_expanded[0])
print("Expended Size:", len(ds_expanded))
