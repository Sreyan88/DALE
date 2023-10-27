#!/bin/sh

python pmi_tokenize_no_tag.py -c $1 -f $2 -b $3 

python k_gram_pmi.py -c $1 -f $2 -b $3 

python pmi_seg.py -c $1 -f $2 -k $4 -b $3 

python pmi_disc.py -c $1 -f $2 -k $4 -b $3 -co $5