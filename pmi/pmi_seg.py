from tqdm.auto import tqdm
import math
import pickle
import time
import argparse
import operator
import os

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
    parser.add_argument('-b', '--base_path', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    config = args.config

    dir_path = os.path.join(args.base_path, config)

    combined_dict = {}
    freq_k = {}
    start = time.time()
    with open(dir_path + "/combined.pkl", "rb") as f:
        combined_dict = pickle.load(f)
        f.close()
    end = time.time()
    print("Time taken for loading combined file : " + str(end-start))

    start = time.time()
    with open(dir_path + "/" + str(args.kgram) + "_gram_norm_freq_dist.pkl", "rb") as f:
        freq_k = pickle.load(f)
        f.close()
    end = time.time()
    print("Time taken for loading {} gram file : {}".format(args.kgram, str(end-start)))

    pmi_n_scores = {}
    for k,v in tqdm(sorted(freq_k.items()), unit="k-Gram", total=len(freq_k), desc="Calculating PMI"):
        segmentations = generate_segmentations(k)
        num = combined_dict[k]
        mine = float('inf')
        for segmentation in segmentations:
            if(len(segmentation) == 1):
                break
            den = 1.0
            for key in segmentation:
                den = den*combined_dict[key]
            mine = min(mine, math.log(num/den))
        # print("{}/{}".format(num, den))
        pmi_n_scores[k] = mine
        # print("{}: {}".format(k, pmi_n_scores[k]))
    
    with open(dir_path + "/" + str(args.kgram) + "_pmi_scores.pkl", "wb") as f:
        pickle.dump(pmi_n_scores, f)


    cnt = 0
    for k,v in sorted(pmi_n_scores.items(), key=operator.itemgetter(1),reverse=True):
        if cnt == 10:
            break
        print("{}: {}".format(k, v))
        cnt = cnt + 1
    
        
            
