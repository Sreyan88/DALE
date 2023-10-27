import operator
from pathlib import Path
from tqdm.auto import tqdm

from datasets import load_from_disk, set_caching_enabled

import os

from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import argparse
import re
from pathlib import Path

def get_top_pmi_ngrams(pmi_path, p, typ): #take p% of top scoring pmi's

    print("--------Extracting top scoring PMIs----------")
    
    all_files = os.listdir(pmi_path)
    
    pmi = {}
    for file in all_files:
        if typ in file:
            print(file)
            with open(pmi_path + "/" + file, 'rb') as pickle_file:
                data = pickle.load(pickle_file)

            if len(data) == 0:
                continue
            data = {" ".join(list(k)):v for k,v in data.items()}

            N = round(p*len(data))
            data = dict(sorted(data.items(), key=operator.itemgetter(1), reverse=True)[:N])

            pmi.update(data)
        
    return pmi

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.\-\_,!?/:;\"\'\s]'
    new_text =  re.sub(pat, '', text)
    return new_text

def clean_pipeline(text):
    return remove_special_characters(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-b', '--base_path', type=str, required=True)
    args = parser.parse_args()

    config = args.config

    dir_path = os.path.join(args.base_path, config)

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.file)

    len_dataset = len(dataset)

    print("Len of dataset : " + str(len(dataset)))

    tokenized_sentences = []

    for text in tqdm(dataset['text'], total=len(dataset['text']), unit="sentence", desc="Tokenizing sentences"):
        text = clean_pipeline(str(text))
        text = text.replace('\n',' ').replace('\t',' ').replace('  ',' ').replace('\'s','').replace('-',' - ')
        sentences = sent_tokenize(text)
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokenized_sentences.append(tokens)
    
    with open(dir_path + "/tokenized_sentences.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)