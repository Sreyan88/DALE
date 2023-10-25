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
import time

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='scotus', help='dataset dir name')
parser.add_argument('--path', type=str, help='dataset dir name')
parser.add_argument('--dest_path', type=str, help='dataset dir name')
parser.add_argument('--multi',type=bool,default=False)
parser.add_argument('--n_augs',type=int,default=5)
parser.add_argument('--model_path', type=str, help='dataset dir name')
args = parser.parse_args()

model_path = args.model_path

dataset = load_from_disk(args.path)

genius = pipeline("text2text-generation", model=model_path, tokenizer=BartTokenizer.from_pretrained(model_path), device=0, max_length=1024)

print(dataset)

gen_dict = defaultdict(list)

ori_dataset = defaultdict(list)

ctx_end_tag = "</context>"

ori_label2desc = get_label2desc(args.dataset_name)
label2desc = {value: key for key, value in ori_label2desc.items()}

label_to_idx_map = {}

for idx, (key, _) in enumerate(ori_label2desc.items()):
    label_to_idx_map[key] = idx

start = time.time()
print(dataset[0])
gen_out = genius((dataset["sketch"]), batch_size=2, num_beams=5, top_k=10, do_sample=True, num_return_sequences=5, max_length=1024)
end = time.time()
print(f"Time taken for generation : {end - start}")
print(f"Len of generated output : {len(gen_out)}")
print(f"Len of element in generated output : {len(gen_out[0])}")
gen_dict = {}

for row, gen_text in zip(tqdm(dataset), gen_out):
    prompt = row["prompt"]
    sketch = row["sketch"].replace(prompt,'')
    context = row["context"]
    id = row["id"]
    if args.multi:
        labels = [label2desc[label] for label in prompt.replace(': ','').split(',')]
    else:
        labels = label2desc[prompt.replace(': ','')]
    if id not in gen_dict:
        gen_dict[id] = {"ori_text": row["text"], "sketch": sketch, "augs":["","","","",""], "gen_text":["","","","",""],"label": labels}
    for idx, generated_text in enumerate(gen_text):
        if len(context) > 0:
            gen_dict[id]["augs"][idx] = gen_dict[id]["augs"][idx] + generated_text["generated_text"][generated_text["generated_text"].find(ctx_end_tag)+len(ctx_end_tag):].replace('<context>', '').replace('</context>', '').replace(prompt, '')
        else:
            gen_dict[id]["augs"][idx] = gen_dict[id]["augs"][idx] + generated_text["generated_text"].replace(prompt, '').replace('<context>', '').replace('</context>', '')
        gen_dict[id]["gen_text"][idx] = gen_dict[id]["gen_text"][idx] + generated_text["generated_text"].replace('<context>', '').replace('</context>', '').replace(prompt, '')

gen_augs = {"text": [], "label": [], "ori_text": [], "sketch": [],"gen_text": []}

for aug_idx in range(5):
    for id, _ in gen_dict.items():
        text = gen_dict[id]["augs"][aug_idx].lstrip()
        if len(text) > 0:
            gen_augs["text"].append(text)
            gen_augs["label"].append(gen_dict[id]["label"])
            gen_augs["ori_text"].append(gen_dict[id]["ori_text"])
            gen_augs["sketch"].append(gen_dict[id]["sketch"])
            gen_augs["gen_text"].append(gen_dict[id]["gen_text"][aug_idx].lstrip())

gen_augs = pd.DataFrame.from_dict(gen_augs)
print(f"Len of generated augs : {len(gen_augs)}")

dest_aug_path = args.dest_path

if not os.path.exists(dest_aug_path):
    os.makedirs(dest_aug_path)

gen_augs.drop('ori_text', axis=1, inplace=True)
gen_augs.drop('gen_text', axis=1, inplace=True)
gen_augs.drop('sketch', axis=1, inplace=True)

combined_augs = gen_augs

combined_augs.to_csv(dest_aug_path + args.dataset_name + "_train.tsv", sep="\t", index=False)

print(f"Len of combined augs : {len(combined_augs)}")