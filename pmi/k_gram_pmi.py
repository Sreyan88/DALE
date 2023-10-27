from tqdm.auto import tqdm
import time

from datasets import load_dataset

import time

import argparse

from nltk.probability import FreqDist
from nltk.util import ngrams
import os
import pickle

def compute_freq(corpus_tokens, n=2):

    ngram_freq_dist = FreqDist()

    for sentence in tqdm(corpus_tokens, total=len(corpus_tokens), desc="Ngram Computation", unit="sentence"):
        ngrams_in_sentence = ngrams(sentence, n)
        ngram_freq_dist.update(ngrams_in_sentence)

    return ngram_freq_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-b', '--base_path', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    config = args.config

    dir_path = os.path.join(args.base_path, config)
    
    total_tokens = 1

    with open(dir_path + "/tokenized_sentences.pkl", "rb") as f:
        tokenized_sentences = pickle.load(f)

    word_norm_freq = {}
    for i in tqdm(range(1,8)):
        print("In index : " + str(i))
        word_k_gram_freq = {}
        word_freq_dist = compute_freq(tokenized_sentences, i)
        if i == 1:
            total_tokens = len(word_freq_dist)
            with open(dir_path + "/total_tokens.pkl", "wb") as f:
                pickle.dump(total_tokens, f)
        for word, count in word_freq_dist.items():
            word_norm_freq[word] = count / total_tokens
            word_k_gram_freq[word] = count / total_tokens

        with open(dir_path + "/" + str(i) + "_gram_norm_freq_dist.pkl", "wb") as f:
            pickle.dump(word_k_gram_freq, f)

    start = time.time()
    with open(dir_path +  "/combined.pkl", "wb") as f:
        pickle.dump(word_norm_freq, f)
    end = time.time()
    
    print("Time taken for saving file : " + str(end-start))