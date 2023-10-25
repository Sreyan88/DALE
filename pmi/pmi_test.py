from tqdm.auto import tqdm
import math
import pickle
import time
import argparse
import operator
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True)
    parser.add_argument('-k', '--kgram', type=int, required=True)
    parser.add_argument('-b', '--base_path', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    config = args.config

    dir_path = os.path.join(args.base_path, config)

    pmi_n_scores = {}

    with open(dir_path + "/total_tokens.pkl", "rb") as f:
        total_tokens = pickle.load(f)
        f.close()
    if args.mode == "pmi_disc":
        with open(dir_path + "/" + str(args.kgram) + "_pmi_disc.pkl", "rb") as f:
            pmi_n_scores = pickle.load(f)
            f.close()
    elif args.mode == "pmi_scores":
        with open(dir_path + "/" + str(args.kgram) + "_pmi_scores.pkl", "rb") as f:
            pmi_n_scores = pickle.load(f)
            f.close()
    else:
        with open(dir_path + "/" + str(args.kgram) + "_gram_norm_freq_dist.pkl", "rb") as f:
            pmi_n_scores = pickle.load(f)
            f.close()      

    pmi_k_gram_freq = {}
    with open(dir_path + "/" + str(args.kgram) + "_gram_norm_freq_dist.pkl", "rb") as f:
        pmi_k_gram_freq = pickle.load(f)
        f.close()

    cnt = 0
    f = open(dir_path + "/" + str(args.kgram) + "_" + args.mode + "_top_10.txt","wb")

    for k, v in sorted(pmi_n_scores.items(), key=operator.itemgetter(1), reverse=True):
        print("{} : {} : {}".format(k, v, total_tokens*pmi_k_gram_freq[k]))
        f.write(str(k).encode('utf-8') + b' : ' + str(v).encode('utf-8') + b' : ' + str((total_tokens*pmi_k_gram_freq[k])).encode('utf-8') + b'\n')
        cnt = cnt + 1
        if cnt == 10:
            break
    f.close()
    #print()
    # print(total_tokens*pmi_n_scores[('nonconventional', 'fuels', 'credit', 'under', 'section', '29', '(', '“', 'Synthetic', 'Fuel4')])