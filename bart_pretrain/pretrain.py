import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_metric
import argparse
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--max_input_length', type=int, required=True)
parser.add_argument('--max_target_length', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--num_train_epochs', type=int, required=True)
parser.add_argument('--logging_steps', type=int, required=True)
parser.add_argument('--save_steps', type=int, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

# pretrained checkpoint:
model_checkpoint = args.ckpt_path  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

##################################################################
#                     data pre-processing
##################################################################

# load the preprocessed dataset with the four kinds of sketches
from datasets import load_from_disk
dataset_path = args.dataset_path
dataset_name = dataset_path.split('/')[-1]
dataset_with_sketch = load_from_disk(dataset_path).select(range(10))
print(dataset_with_sketch)

# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = args.max_input_length
max_target_length = args.max_target_length

def preprocess_function(examples):
    """
    # inputs: the sketch
    # labels: the original text
    """
    model_inputs = tokenizer(examples['mask'], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset_with_sketch.map(preprocess_function, batched=True, 
                                         batch_size=10000,num_proc=25)


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

# load the pretrained weights
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    do_eval=False,
    save_strategy = 'steps',
    save_steps=args.save_steps,
    save_total_limit = 2,
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=args.batch_size,
    weight_decay=0.01,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True,
    logging_steps=args.logging_steps,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(dataset_with_sketch.column_names)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()