import sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, RobertaTokenizer, PreTrainedTokenizerFast
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
parser.add_argument('--path', type=str, help='dataset dir name', required=True)
parser.add_argument('--split', type=int, default=100, help='', required=True)
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


# read dataset
dataset = pd.read_csv(args.path.format(args.split, args.dataset_name), sep="\t", header=0)
contents = list(dataset['text'])
labels = list(dataset['label'])
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

def my_topk(text):
    l = len(text.split(' '))
    if args.type == "SM":
        return min(max(l//5, 1), 40)
    else:
        return min(max(l//5, 1), 60)

print('Extracting chunks and sketches...')
stopwords = get_stopwords()
sketches = []
output_texts = []
contexts = []
prompts = []
ids = []
print("Extracting sketches.")
for idx, dict in enumerate(tqdm(new_dataset, total=len(new_dataset))):
    for i in range(len(dict["original"])):
        label = dict["label"]
        content = dict["original"][i]
        prompt = label2desc[label] + ': '
        aspect_keywords = label2desc[label].split(' ')
        topk = my_topk(content)
        if len(dict["context"][i].strip()) > 0:
            kws = sketcher.get_kws(
                    dict["context"][i] + content, min_ngrams=args.min_ng, max_ngram=args.max_ng,top=max(2,topk), 
                    aspect_keywords=aspect_keywords, 
                    use_aspect_as_doc_embedding=args.aspect_only
                )[1]
            kws = [w for w in kws if w not in stopwords]
            sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
            sketch = prompt + "<context>" + dict["context"][i] + "</context>" + sketch
            sketches.append(sketch)
            output_texts.append(prompt + "<context>" + dict["context"][i] + "</context>" + content)
            contexts.append(prompt + dict["context"][i])
        else:
            kws = sketcher.get_kws(
                    content, min_ngrams=args.min_ng, max_ngram=args.max_ng, top=max(2,topk) , 
                    aspect_keywords=aspect_keywords, 
                    use_aspect_as_doc_embedding=args.aspect_only, 
                )[1]
            kws = [w for w in kws if w not in stopwords]
            sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
            sketch = prompt + sketch
            sketches.append(sketch)
            output_texts.append(prompt + content)
            contexts.append("")            
        prompts.append(prompt)
        ids.append(idx)
sketch_dataset = List2Dataset(sketches)

text_dataset = HFDataset.from_dict({'context': contexts, 'sketch':sketch_dataset, 'text':output_texts, 'prompt': prompts, "id":ids})
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