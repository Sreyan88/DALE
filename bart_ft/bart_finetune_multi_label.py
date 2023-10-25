import sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from genius_utils import SketchExtractor, List2Dataset, get_stopwords
from datasets import load_metric
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
random.seed(5)
import argparse
import torch
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, help='dataset dir name', required=True)
parser.add_argument('--path', type=str, help='dataset dir name')
parser.add_argument('--split', type=int, default=100, help='')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--aspect_only', action='store_true', default=False, help='')
parser.add_argument('--template', type=int, default=4, help='')
parser.add_argument('--num_train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for dataloaders')
parser.add_argument('--max_num_sent', type=int, default=15, help='num of sentences for a chunk')
parser.add_argument('--comment', type=str, default='', help='to modify save name')
parser.add_argument('--min_ng',type=int,default=1)
parser.add_argument('--max_ng',type=int,default=3)
parser.add_argument('--type',type=bool,default="SM")
parser.add_argument('--model',type=str, required=True)
parser.add_argument('--tag',type=str,default="")
args = parser.parse_args()

def_label = {
    "ecthr_a": "No violations",
    "ecthr_b": "No violations",
    "unfair_tos": "Fair",
    "ots_topics": "Fair"
}

col_to_idx_map = {
    "ecthr_a" : {
        "1": 2,
        "2": 3,
        "3": 5,
        "4": 6,
        "5": 8,
        "6": 9,
        "7": 10,
        "8": 11,
        "9": 14,
        "10": "P1-1"
    },
    "ecthr_b" : {
        "1": 2,
        "2": 3,
        "3": 5,
        "4": 6,
        "5": 8,
        "6": 9,
        "7": 10,
        "8": 11,
        "9": 14,
        "10": "P1-1"
    },
    "eurlex" : {
        "1": "100163", "2": "100168", "3": "100169", "4": "100170", "5": "100171", "6": "100172", "7": "100173", "8": "100174", "9": "100175", "10": "100176", "11": "100177", "12": "100179", "13": "100180", "14": "100183", "15": "100184", "16": "100185", "17": "100186", "18": "100187", "19": "100189", "20": "100190", "21": "100191", "22": "100192", "23": "100193", "24": "100194", "25": "100195", "26": "100196", "27": "100197", "28": "100198", "29": "100199", "30": "100200", "31": "100201", "32": "100202", "33": "100204", "34": "100205", "35": "100206", "36": "100207", "37": "100212", "38": "100214", "39": "100215", "40": "100220", "41": "100221", "42": "100222", "43": "100223", "44": "100224", "45": "100226", "46": "100227", "47": "100229", "48": "100230", "49": "100231", "50": "100232", "51": "100233", "52": "100234", "53": "100235", "54": "100237", "55": "100238", "56": "100239", "57": "100240", "58": "100241", "59": "100242", "60": "100243", "61": "100244", "62": "100245", "63": "100246", "64": "100247", "65": "100248", "66": "100249", "67": "100250", "68": "100252", "69": "100253", "70": "100254", "71": "100255", "72": "100256", "73": "100257", "74": "100258", "75": "100259", "76": "100260", "77": "100261", "78": "100262", "79": "100263", "80": "100264", "81": "100265", "82": "100266", "83": "100268", "84": "100269", "85": "100270", "86": "100271", "87": "100272", "88": "100273", "89": "100274", "90": "100275", "91": "100276", "92": "100277", "93": "100278", "94": "100279", "95": "100280", "96": "100281", "97": "100282", "98": "100283", "99": "100284", "100": "100285"
    },
    "unfair_tos" : {
        "1": 0, "2" : 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7
    },
    "ots_topics" : {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10
    }
}

# read dataset
dataset = pd.read_csv(args.path.format(args.split, args.dataset_name), sep="\t", header=0)
contents = list(dataset["0"])
print(f"Len of contents : {len(contents)}")
labels = []
for idx, row in dataset.iterrows():
    label = []
    for col in range(1,dataset.shape[1]):
        if row[str(col)] == 1:
            label.append(str(col))
    labels.append(label)

# use the label names/descriptions to replace the label indices
from label_mapping.label_desc import get_label2desc
if get_label2desc(args.dataset_name):
    label2desc = get_label2desc(args.dataset_name)
    print(label2desc)
else:
    label2desc = {label:label for label in set(labels)}

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.add_tokens(['<context>', '</context>'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    # label tokens
    model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[46796, :].clone()
    model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[46796, :].clone()
    model.model.decoder.embed_tokens.weight[-1, :] += model.model.decoder.embed_tokens.weight[46796, :].clone()
    model.model.decoder.embed_tokens.weight[-2, :] += model.model.decoder.embed_tokens.weight[46796, :].clone()

sketcher = SketchExtractor(model='bert')

def generate_new_list(paragraph, max_length=900):
    max_length = max_length - 15
    tokens = tokenizer.tokenize(paragraph)
    if(len(tokens)) <= max_length:
        return {"original":[paragraph], "context":[""]}

    sentences = sent_tokenize(paragraph)
    
    tot = len(sentences)
    cnt = 0
    new_list = {"original":[],"context":[]}
    previous_list = []
    token_len_list = [-1]*tot
    current_length = 0
    isContext = False
    context_list = []
    while cnt < tot:
        sentence = sentences[cnt]
        tokenized_sentence = tokenizer.tokenize(sentence)
        sentence_length = len(tokenized_sentence)
        token_len_list[cnt] = sentence_length

        if sentence_length > max_length:
            if len(previous_list) > 0:
                new_list["original"].append(" ".join(previous_list))
                new_list["context"].append("")
            new_list["original"].append(sentence)
            new_list["context"].append("")
            isContext = False
            context_list = []
            previous_list = []
            current_length = 0
        elif current_length + sentence_length <= max_length:
            previous_list.append(sentence)
            current_length += sentence_length
        else:
            new_list["original"].append(" ".join(previous_list))
            if isContext:
                new_list["context"].append(" ".join(context_list))
                context_list = []
                isContext = False
            else:
                new_list["context"].append("")
                isContext = False
            if cnt >= 5:
                p = 0.2
                while(True and p>0):
                    temp = cnt - int(p*(len(previous_list)))
                    if (sum(token_len_list[temp:cnt+1])) <= max_length:
                        isContext = True
                        context_list = context_list + (sentences[temp:cnt])
                        break
                    else:
                        p = p - 0.05
                temp = cnt - int(p*len(previous_list))

                if temp == cnt:
                    new_list["original"].append(" ".join(previous_list))
                    new_list["context"].append("")
                    previous_list=[]
                    context_list=[]
                    current_length = 0
                else:
                    previous_list = [sentence]
                    current_length = sum(token_len_list[temp:cnt+1])

        cnt = cnt + 1

    if len(previous_list)>0:
        new_list["original"].append(" ".join(previous_list))
        if isContext:
            new_list["context"].append(" ".join(context_list))
            isContext = False
        else:
            new_list["context"].append("")

    return new_list

new_dataset = []
print("Creating original and context paras.")
for content, label in zip(tqdm(contents, total=len(contents), unit="Paragraphs", desc="Parsing paras"), labels):
    dataset_dict = generate_new_list(content, 900)
    dataset_dict["label"] = label
    new_dataset.append(dataset_dict)

print(f"Len of new dataset is {len(new_dataset)}")

print(new_dataset[0])

def my_topk(text):
    l = len(text.split(' ')) 
    if args.type == "SM":
        return min(max(l//5, 1), 40)
    else:
        return min(max(l//5, 1), 60)

print('Extracting chunks and sketches...')
stopwords = get_stopwords()
sketches = []
contexts = []
output_texts = []
prompts = []
ids = []
print("Extracting sketches.")
for idx, dict in enumerate(tqdm(new_dataset, total=len(new_dataset))):

    for i in range(len(dict["original"])):
        label_list = dict["label"]
        label_str= ""
        len_labels = len(label_list)
        if len_labels > 0:
            for k, id in enumerate(label_list):
                label_str = label_str + label2desc[col_to_idx_map[args.dataset_name][id]]
                if k != len_labels - 1:
                    label_str = label_str + ","
        else:
            label_str = def_label[args.dataset_name]
        content = dict["original"][i]
        prompt = label_str + ': '
        aspect_keywords = label_str.replace(',',' ').split(' ')

        topk = my_topk(content)

        if len(dict["context"][i].strip()) > 0:
            kws = sketcher.get_kws(
                    dict["context"][i] + content, min_ngrams=args.min_ng, max_ngram=args.max_ng, top=max(2,topk), 
                    aspect_keywords=aspect_keywords, 
                    use_aspect_as_doc_embedding=args.aspect_only, 
                )[1]
            kws = [w for w in kws if w not in stopwords]            
            sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
            if sketch == '':
                sketch = content
            sketch = prompt + "<context>" + dict["context"][i] + "</context>" + sketch
            sketches.append(sketch)
            output_texts.append(prompt + "<context>" + dict["context"][i] + "</context>" + content)
            contexts.append(prompt + dict["context"][i])
        else:
            kws = sketcher.get_kws(
                    content, min_ngrams=args.min_ng, max_ngram=args.max_ng, top=max(2,topk), 
                    aspect_keywords=aspect_keywords, 
                    use_aspect_as_doc_embedding=args.aspect_only, 
                )[1]
            kws = [w for w in kws if w not in stopwords]
            sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
            if sketch == '':
                sketch = content
            sketch = prompt + sketch
            sketches.append(sketch)
            output_texts.append(prompt + content)
            contexts.append("")          
        prompts.append(prompt)
        ids.append(idx)

sketch_dataset = List2Dataset(sketches)
print(f"Len of sketches :{len(sketch_dataset)}")
text_dataset = HFDataset.from_dict({'context': contexts, 'sketch':sketch_dataset, 'text':output_texts, 'prompt': prompts, "id":ids})
print(f"Len of text dataset :{len(text_dataset)}")
print(f"Text dataset ids : {text_dataset['id']}")
text_dataset.save_to_disk(f"../bart_ft_datasets/{args.dataset_name + '_' + args.tag}/{args.split}")
print("Saved sketch dataset to disk")
# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = 1024
max_target_length = 1024
print("********** Sketch type is: ", args.template)
def preprocess_function(examples):
    """
    # inputs: the sketch
    # labels: the original text
    """
    model_inputs = tokenizer(examples['sketch'], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = text_dataset.map(preprocess_function, batched=True)

# ROUGE metric：
rouge_score = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


##################################################################
#                     training
##################################################################

output_dir = f"../saved_models/bart_finetuned_for_{args.dataset_name + '_' + args.tag}/{args.split}/"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy = 'no', # maybe set to 'no' to save time?
    save_total_limit = 1,
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset.remove_columns(text_dataset.column_names),
    eval_dataset=tokenized_dataset.remove_columns(text_dataset.column_names), # just look at the train set
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train(resume_from_checkpoint = False)
trainer.save_model(output_dir)