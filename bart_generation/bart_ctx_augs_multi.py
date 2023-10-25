import sys
sys.path.append('../')
from collections import defaultdict
from transformers import pipeline, BartTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_from_disk, Dataset
import argparse
import pandas as pd
from tqdm import tqdm
from label_mapping.label_desc import get_label2desc
import os

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='unfair_tos', help='dataset dir name')
parser.add_argument('--path', type=str, help='dataset dir name')
parser.add_argument('--dest_path', type=str, help='dataset dir name')
parser.add_argument('--multi',type=bool,default=True)
parser.add_argument('--n_augs',type=int,default=5)
parser.add_argument('--model_path', type=str, help='dataset dir name')
args = parser.parse_args()

label_size = {
    "unfair_tos": 8,
    "eurlex": 100,
    "ecthr_a": 10,
    "ecthr_b": 10,
    "ots_topics": 9,
    "unfair_tos_fb": 8,
    "eurlex_fb": 100,
    "ecthr_a_fb": 10,
    "ecthr_b_fb": 10,
    "ots_topics_fb": 9
}

model_path = args.model_path

dataset = load_from_disk(args.path)

genius = pipeline("text2text-generation", model=model_path, tokenizer=BartTokenizer.from_pretrained(model_path), device=0, max_length=1024)

gen_dict = defaultdict(list)

ori_dataset = defaultdict(list)

ctx_end_tag = "</context>"

ori_label2desc = get_label2desc(args.dataset_name)
label2desc = {value: key for key, value in ori_label2desc.items()}

label_to_idx_map = {}

for idx, (key, _) in enumerate(ori_label2desc.items()):
    label_to_idx_map[key] = str(idx+1)

print(label_to_idx_map)

print(dataset)

gen_out = genius((dataset["sketch"]), batch_size=4, num_beams=5, top_k=10, do_sample=True, num_return_sequences=5, max_length=500)
print(f"Len of generated output : {len(gen_out)}")
print(f"Len of element in generated output : {len(gen_out[0])}")

gen_dict = {}

print(f"Dataset ids : {dataset['id']}")

for row, gen_text in zip(tqdm(dataset), gen_out):
    prompt = row["prompt"]
    sketch = row["sketch"].replace(prompt,'')
    context = row["context"]
    id = row["id"]
    labels = [label_to_idx_map[label2desc[label]] for label in prompt.replace(': ','').split(',')]
    if id not in gen_dict:
        gen_dict[id] = {"ori_text": row["text"], "sketch": sketch, "augs":["","","","",""], "gen_text":["","","","",""],"label": labels}
    for idx, generated_text in enumerate(gen_text):
        if len(context) > 0:
            gen_dict[id]["augs"][idx] = gen_dict[id]["augs"][idx] + generated_text["generated_text"][generated_text["generated_text"].find(ctx_end_tag)+len(ctx_end_tag):].replace('<context>', '').replace('</context>', '').replace(prompt, '')
        else:
            gen_dict[id]["augs"][idx] = gen_dict[id]["augs"][idx] + generated_text["generated_text"].replace(prompt, '').replace('<context>', '').replace('</context>', '')
        gen_dict[id]["gen_text"][idx] = gen_dict[id]["gen_text"][idx] + generated_text["generated_text"].replace('<context>', '').replace('</context>', '').replace(prompt, '')
    # print(f"Len of gen dict augs : {len(gen_dict[id]['augs'])}")

print(f"Len of gen_dict : {len(gen_dict)}")

print(f"Keys : {gen_dict.keys()}")

gen_augs = {"0": [], "ori_text": [], "sketch": [],"gen_text": []}

for i in range(1, label_size[args.dataset_name] + 1):
    gen_augs[str(i)] = []

count = 0

for aug_idx in range(5):
    for id, _ in gen_dict.items():
        count += 1
        text = gen_dict[id]["augs"][aug_idx].lstrip()
        # print(f"Labels for id : {id} are {gen_dict[id]['label']}")
        if len(text) > 0:
            gen_augs["0"].append(text)
            for i in range(1, label_size[args.dataset_name] + 1):
                if str(i) in gen_dict[id]["label"]:
                    gen_augs[str(i)].append(1)
                else:
                    gen_augs[str(i)].append(0)
                # gen_augs[str(i)].append(ori_data[str(i)][int(id)])
            gen_augs["ori_text"].append(gen_dict[id]["ori_text"])
            gen_augs["sketch"].append(gen_dict[id]["sketch"])
            gen_augs["gen_text"].append(gen_dict[id]["gen_text"][aug_idx].lstrip())
        else:
            print(f"Empty string for id : {id} and text is : {text}")

print(f"Total count is {count}")

gen_augs = pd.DataFrame.from_dict(gen_augs)
print(f"Len of generated augs : {len(gen_augs)}")
# print(gen_augs)
dest_aug_path = args.dest_path

if not os.path.exists(dest_aug_path):
    os.makedirs(dest_aug_path)

# gen_augs.to_csv(dest_aug_path + args.dataset_name + "_int_train.tsv", sep="\t", index=False)

gen_augs.drop('ori_text', axis=1, inplace=True)
gen_augs.drop('gen_text', axis=1, inplace=True)
gen_augs.drop('sketch', axis=1, inplace=True)

combined_augs = gen_augs

combined_augs.to_csv(dest_aug_path + args.dataset_name + "_train.tsv", sep="\t", index=False)

print(f"Len of combined augs : {len(combined_augs)}")