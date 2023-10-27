from collections import defaultdict
from tqdm.auto import tqdm
from datasets import load_from_disk, Dataset
import os
from nltk.tokenize import sent_tokenize
import pickle
import argparse
import pickle
import os
import torch.nn.functional as F
import spacy
from transformers import AutoTokenizer, AutoModel
import operator
import torch
import networkx as nx
import textacy
from tqdm.auto import tqdm
import re
import random

nlp = spacy.load("en_core_web_lg")
tokenizer = AutoTokenizer.from_pretrained("pile-of-law/legalbert-large-1.7M-1")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModel.from_pretrained("pile-of-law/legalbert-large-1.7M-1").cuda()
model.eval()

def get_sentences(base_path, dataset, debug=False):
    if debug ==True:
        dataset = load_from_disk(os.path.join(base_path,dataset)).select(range(10))
    else:
        dataset = load_from_disk(os.path.join(base_path, dataset))
    print(dataset)

    tokenized_sentences = []
    for text in tqdm(dataset['text'], total=len(dataset['text']), unit="sentence", desc="Tokenizing sentences"):
        text = clean_text(text)
        sentences = sent_tokenize(text)
        tokenized_sentences.append(sentences)
            
    return tokenized_sentences

def find_top_k_sentences(sentences):
    # don't take more than 512 sentences in a paragraph
    sentences = sentences[:512]
    # take only 512 tokens for each sentence, through truncation (handled by BERT)
    encoded_inputs = tokenizer.batch_encode_plus(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(encoded_inputs['input_ids'].cuda(),encoded_inputs['attention_mask'].cuda()).last_hidden_state.detach().cpu()

    cls_embeddings = outputs[:,0,:]
    similarity_matrix = torch.mm(cls_embeddings, cls_embeddings.T).numpy()
    G = nx.DiGraph(similarity_matrix)
    pagerank_scores = nx.pagerank(G)

    if random.gauss(0.5, 0.1) > 0.45:
        sorted_sentences = sorted(sentences, key=lambda s: -pagerank_scores[sentences.index(s)])
    else:
        sorted_sentences = sentences.copy()
    
    top_k_sentences = []
    length = 0
    # remember this disrupts sentence order
    for i,sentence in enumerate(sorted_sentences):
        # record the length, -2 to ignore CLS and SEP tokens
        bart_length = len(bart_tokenizer.encode(sorted_sentences[i])) - 2
        # pick sentences according to length, total token length should not exceed 1024 (BART has 1024 length limit)
        if (length + bart_length) <= 1010:
            top_k_sentences.append(sorted_sentences[i])
            length += bart_length
        
    # sort sentences back to original order
    top_k_sentences = sorted(top_k_sentences, key=lambda s: sentences.index(s))
        
    return top_k_sentences

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

def get_encoding(sentences):

    # Truncation will already truncate sentences to what BERT can take
    encoded_inputs = tokenizer(' '.join(sentences), max_length=512, padding=True, truncation=True, return_tensors = 'pt')
    # encoded_inputs = tokenizer.batch_encode_plus(sentences, padding=True, return_tensors='pt')

    with torch.no_grad():
        outputs_sentences_with_para = model(encoded_inputs['input_ids'].cuda()).last_hidden_state.detach().cpu()

    cls_embeddings_para = outputs_sentences_with_para[:,0,:]

    return cls_embeddings_para

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.\-\_,!?/:;\"\'\s]'
    new_text =  re.sub(pat, '', text)
    return new_text

def clean_pipeline(text):
    return remove_special_characters(text)

def clean_text(text):
    text = clean_pipeline(text)
    text = text.replace('\n',' ').replace('\t',' ').replace('  ',' ').replace('\'s','').replace('-',' - ')
    return text

def remove_consecutive_strings(lst):
    result = []
    prev = None
    mask_found = False
    for s in lst:
        if s == '<mask>':
            if not mask_found:
                result.append(s)
                mask_found = True
            prev = None
        elif s != prev:
            result.append(s)
            prev = s
            mask_found = False
    return result

def find_overlapping_keys(key, dictionary):
    # How do we relax the constraint?
    overlapping_keys = []
    key_range = dictionary[key] # a list, 1st item PMI value, 2nd item is a list of tuples
    # key_range[1] is the list of tuples of start and end indices (if a single n-gram has multiple occurences)
    for l in range(len(key_range[1])):
        key_range_single = key_range[1][l] # a particular key range, because one n-gram might have multiple ranges
        for k, v in dictionary.items():
            if k != key: # starting anywhere between the start and end of the span --> #ending anywhere between the start and end of the span
                for e in range(len(v[1])): # the aim is to find all overlapping spans with the span to keep, v[1][e] is a particular key range
                    overlap_spans_indices = [] # 
                    if ((key_range_single[0] <= v[1][e][0]) and (key_range_single[1] >= v[1][e][0])) or ((key_range_single[0] <= v[1][e][1]) and (key_range_single[1] >= v[1][e][1])):
                        overlap_spans_indices.append(v[1][e]) # first store all spans which overlap, to not take all occurences of a span as overlap
                # now store all spans
                if len(overlap_spans_indices) > 0:
                    overlapping_keys.append((k,overlap_spans_indices))
    return overlapping_keys

def mask_and_save(base_path, dataset, pmi_path, output_path, typ, debug):
    
    # get tokenized paragraphs or training instances
    paragraphs = get_sentences(base_path, dataset, debug)
    
    # get top k PMIs
    pmi = get_top_pmi_ngrams(pmi_path, 0.5, typ)

    hf_dataset_dict = defaultdict(list)
    
    for i,sentences in tqdm(enumerate(paragraphs), total=len(paragraphs), unit="para"):

        if debug==True and i==10:
            break

        # find top k sentences
        sentences_len = len(sentences)
        if debug ==True:
            print("Len of sentences is {}".format(sentences_len))
        if sentences_len == 0:
            continue
        sentences = find_top_k_sentences(sentences)

        # find average length of sentences in the top k sentences
        # average_length_top_k = np.mean([len(word_tokenize(sentence)) for sentence in sentences]) #add word tokenize logic from Chandra
        # find max size of ngrams to "keep". These ngrams will act as hints.
        # ngram_end_nomask =  round(0.5 * average_length_top_k)
        ngram_end_nomask =  4
        # find max size of ngrams to "remove".
        ngram_end = ngram_end_nomask + 3

        cls_embeddings_sentences = get_encoding(sentences)

        # join all sentences to make a paragraph
        paragraph = " ".join(sentences)

        # initialize a sentence to be masked and pass it through nlp
        spacy_sentences = nlp(paragraph)

        # updated_sentences = word_tokenize(clean_text(paragraph)) #add word tokenize logic from Chandra
        updated_sentences = [str(spacy_sentences[i]) for i in range(len(spacy_sentences))]

        #make a copy of this as it is better you save this list as final target (since you are masking using this)
        final_target_sentences = updated_sentences.copy()

        available_ngrams = {}
        # start_end = []

        # first find ngrams in each sentence
        for n_gram in range(2,ngram_end+1):

            ngrams_in_sentence = textacy.extract.ngrams(spacy_sentences, n_gram, filter_stops = False, filter_punct= False)

            for ngram in ngrams_in_sentence:
                if str(ngram) in pmi:
                    if str(ngram) in available_ngrams:
                        available_ngrams[str(ngram)][1].append((ngram.start,ngram.end))
                    else:
                        available_ngrams[str(ngram)] = [pmi[str(ngram)],[(ngram.start,ngram.end)]]

        num_ngrams_available = len(available_ngrams)
        # proceed only if n-grams are available
        if num_ngrams_available > 0:
            # tokenize and pass ngrams through model
            tokenized_available_ngrams = tokenizer.batch_encode_plus(list(available_ngrams.keys()), padding=True, return_tensors = 'pt')
            with torch.no_grad():
                outputs_available_ngrams = model(tokenized_available_ngrams['input_ids'].cuda(),tokenized_available_ngrams['attention_mask'].cuda()).last_hidden_state.detach().cpu()

            # cls embeddings of all available n_grams
            cls_embeddings_available_ngrams = outputs_available_ngrams[:,0,:]

            # cosine similarity between sentence embedding and ngrams
            # cosine similarity = normalize the vectors & multiply
            C = F.normalize(cls_embeddings_sentences) @ F.normalize(cls_embeddings_available_ngrams).T

            # sort n-grams according to importance
            N = 0.2*len(updated_sentences)
            ngrams_list_temp = list(available_ngrams.keys())
            C = C.numpy().tolist()
            sorted_ngrams = {x:y for y,x in sorted(zip(C[0], ngrams_list_temp))}

            # compulsorily keep the top scoring ngrams without any overlap (below the ngram_end_nomask length threshold)
            to_remove_from_available_ngrams = []
            to_keep_length = 0
            for n_gram in sorted_ngrams:
                len_ngram = len(n_gram.split(" "))
                # check if you are exceeding a maximum of N words
                if to_keep_length <= N:
                    # check if "n" of n-gram exceeds what you want it to exceed, calculated earlier
                    if len_ngram <= ngram_end_nomask:
                        if len(to_remove_from_available_ngrams) > 0:
                            n_gram_keys = [x[0] for x in to_remove_from_available_ngrams]
                        else:
                            n_gram_keys = []
                        if n_gram not in n_gram_keys: # check if already present (already identified or in overlaps)
                            to_remove_from_available_ngrams.append((n_gram,available_ngrams[n_gram][1]))
                            # now also find overlaps
                            to_remove_from_available_ngrams.extend(find_overlapping_keys(n_gram, available_ngrams))
                            to_keep_length += len_ngram

            # now remove them from the dictionary - "of all available ngrams"
            for ng in to_remove_from_available_ngrams:
                key_to_remove = ng[0] # 0th item of the tuple
                values_to_remove = ng[1] # 1st item of the tuple
                for tupl_e in values_to_remove:
                    try:
                        available_ngrams[key_to_remove][1].remove(tupl_e) # remove the start_end indices from the tuple
                    except:
                        pass

                # if key has no values just delete the key
                try:
                    if len(available_ngrams[key_to_remove][1]) == 0:
                        del available_ngrams[key_to_remove]
                except:
                    pass
            
            to_remove = []
            to_remove_check = []
            # available_ngrams = dict(sorted(available_ngrams.items(), key=lambda x:(x[1], -x[1][1][1]))) #check if this sorts twice
            available_ngrams = dict(sorted(available_ngrams.items(), key=lambda x:len(x[0]),reverse=True))

            # Finalize if you want to keep this
            for item in available_ngrams:
                if len(to_remove_check) > 0:
                    to_remove_check_temp = [k[0] for k in to_remove_check]
                else:
                    to_remove_check_temp = []
                if item not in to_remove_check_temp:
                    to_remove.append((item,available_ngrams[item][1]))
                    to_remove_check.append((item,available_ngrams[item][1]))
                    to_remove_check.extend(find_overlapping_keys(item,available_ngrams)) # improve this logic

            # flatten the list you want to remove
            to_remove_final = []
            # to_remove_final_words = []
            for to_r in to_remove:
                spans = to_r[1]
                to_remove_final.extend(spans)

            # remove 5% of the tuples randomly
            random_tuples_from_final = random.sample(to_remove_final,int(0.05*len(to_remove_final)))
            for rand_tup in random_tuples_from_final:
                to_remove_final.remove(rand_tup)

            # Replace words within the ranges with '<mask>'
            indices_masked = []
            for start, end in to_remove_final:
                for i in range(start, end):
                    # drop words to be masked by 10% probability
                    if random.gauss(0.5, 0.1) > 0.35:
                        updated_sentences[i] = '<mask>'
                        indices_masked.append(i)

            # now mask 10% of random words
            inidices_not_masked = list(set([i for i in range(len(updated_sentences))]).difference(set(indices_masked)))
            random_percent = int(0.1 * len(inidices_not_masked))
            random_indices = random.sample(inidices_not_masked, random_percent)

            for rand_ind in random_indices:
                updated_sentences[rand_ind] = '<mask>'

            # remove consecutive masks
            updated_sentences = remove_consecutive_strings(updated_sentences)

        hf_dataset_dict['text'].append(' '.join(final_target_sentences))
        hf_dataset_dict['mask'].append(' '.join(updated_sentences))

    hf_dataset = Dataset.from_dict(hf_dataset_dict)
    if debug == True:
        hf_dataset.save_to_disk(os.path.join(output_path, dataset))
    else:
        hf_dataset.save_to_disk(os.path.join(output_path, dataset))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-n", "--name", type = str, help="dataset name")
    parser.add_argument( "-p", "--pmi_path", type = str, help="pmi folder path")
    parser.add_argument("-d", "--debug", type=bool, help="debug flag")
    parser.add_argument('-b', '--base_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    base_path = args.base_path
    dataset = args.name
    pmi_path = args.pmi_path
    output_path = args.output_path
    typ = 'disc.pkl'
    print("For dataset : {}".format(dataset))
    mask_and_save(base_path, dataset, pmi_path, output_path, typ, args.debug)