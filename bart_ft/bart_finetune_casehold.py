import ast
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
parser.add_argument('--dataset_name', type=str, default='case_hold', help='dataset dir name')
parser.add_argument('--path', type=str, help='dataset dir name', required=True)
parser.add_argument('--split', type=int, default=100, help='')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--aspect_only', action='store_true', default=False, help='')
parser.add_argument('--template', type=int, default=4, help='')
parser.add_argument('--num_train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for dataloaders')
parser.add_argument('--max_num_sent', type=int, default=15, help='num of sentences for a chunk')
parser.add_argument('--comment', type=str, default='', help='to modify save name')
parser.add_argument('--model',type=str, required=True)
parser.add_argument('--tag',type=str,default="")
args = parser.parse_args()


# read dataset
dataset = pd.read_csv(args.path.format(args.split, args.dataset_name), sep="\t", header=0)
contents = list(dataset['context'])
endings = list(dataset['endings'])
labels = list(dataset['label'])
# use the label names/descriptions to replace the label indices
label2desc = {}
for idx, row in dataset.iterrows():
    label2desc[idx] = ast.literal_eval(row["endings"])[row["label"]]

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.add_tokens(['<HOLDING>'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    # label tokens
    model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[25176, :].clone()
    model.model.decoder.embed_tokens.weight[-1, :] += model.model.decoder.embed_tokens.weight[25176, :].clone()

sketcher = SketchExtractor(model='bert')

def my_topk(text):
    l = len(text.split(' '))
    return min(max(l//5, 1),40)

print('Extracting chunks and sketches...')
stopwords = get_stopwords()
sketches = []
output_texts = []
prompts = []
print("Extracting sketches.")
for idx, content in enumerate(tqdm(contents, total=len(contents))):
    cnt_idx = content.find("(<HOLDING>)")
    if cnt_idx == -1:
        cnt_idx = len(content)
        content = content + " (<HOLDING>)"
    prompt = label2desc[idx] + ': '
    aspect_keywords = label2desc[idx].split(' ')
    
    topk = my_topk(content[:cnt_idx])

    kws = sketcher.get_kws(
            content[:cnt_idx], max_ngram=3,top=topk, 
            aspect_keywords=aspect_keywords, 
            use_aspect_as_doc_embedding=args.aspect_only, 
        )[1]
    kws = [w for w in kws if w not in stopwords]
    sketch = sketcher.get_sketch_from_kws(content[:cnt_idx], kws, template=args.template)
    sketch = prompt + sketch + " " + content[cnt_idx:]
    sketches.append(sketch)
    prompts.append(prompt)
    output_texts.append(prompt + content)
sketch_dataset = List2Dataset(sketches)

text_dataset = HFDataset.from_dict({'sketch':sketch_dataset, 'text':output_texts, 'prompt': prompts, 'endings': endings,'label': labels})
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