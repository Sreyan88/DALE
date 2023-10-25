from tqdm.auto import tqdm
import math
import pickle
import argparse
import os
import numpy as np
import time

def generate_segmentations(tup):
    n = len(tup)
    if n == 1:
        yield (tup,)
    else:
        for i in range(1, n):
            for segment in generate_segmentations(tup[i:]):
                yield ((tup[:i],) + segment)
        yield (tup,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-k', '--kgram', type=int, required=True)
    parser.add_argument('-co', '--cut_off', type=int, required=True)
    parser.add_argument('-b', '--base_path', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    config = args.config

    dir_path = os.path.join(args.base_path, config)

    pmi_scores = {}
    freq_k = {}

    start = time.time()
    with open(dir_path + "/" + str(args.kgram) + "_pmi_scores.pkl", "rb") as f:
        pmi_scores = pickle.load(f)
        f.close()
    end = time.time()
    print("Time taken for loading {} gram pmi_scores file : {}".format(args.kgram, str(end-start)))

    start = time.time()
    with open(dir_path + "/" + str(args.kgram) + "_gram_norm_freq_dist.pkl", "rb") as f:
        freq_k = pickle.load(f)
        f.close()
    end = time.time()
    print("Time taken for loading {} gram freq file : {}".format(args.kgram, str(end-start)))

    with open(dir_path + "/total_tokens.pkl", "rb") as f:
        total_tokens = pickle.load(f)
        f.close()
    print("Total tokens : {}".format(total_tokens))

    values = list((freq_k.values()))

    c = np.percentile(values, args.cut_off)

    logc = math.log((c*total_tokens))

    print("First quartile {} log value : {} freq value : {}".format(c, logc, c*total_tokens))

    for k in list(freq_k.keys()):
        if freq_k[k]<=c:
            freq_k.pop(k)

    pmi_disc = {}
    disc_fact = {}
    for k,v in tqdm(freq_k.items(), unit="K-Gram", total=len(freq_k), desc="Calculating Disc PMI"):
        if freq_k[k] >= c:
            log_freq = math.log(freq_k[k]*total_tokens)
            if (logc + log_freq) == 0.0:
                disc_fact[k] = 0.0
            else:
                disc_fact[k] = log_freq/(logc + log_freq)
            pmi_disc[k] = pmi_scores[k]*disc_fact[k]
    
    with open(dir_path  + "/" + str(args.kgram) + "_pmi_disc.pkl", "wb") as f:
        pickle.dump(pmi_disc, f)