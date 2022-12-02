import os
import numpy as np
import pandas as pd
import json
import collections

def main():
    meta_dir = 'datasets/referit3d/annotations/meta_data'

    data = pd.read_csv(os.path.join(meta_dir, 'scannetv2-labels.combined.tsv'), sep='\t', header=0)
    categories = list(data['raw_category'])
    print('#cat', len(categories))

    json.dump(
        categories, 
        open(os.path.join(meta_dir, 'scannetv2_raw_categories.json'), 'w'),
        indent=2
    )

    uniq_words = collections.Counter()
    for x in categories:
        for w in x.strip().split():
            uniq_words[w] += 1
    print('#uniq words', len(uniq_words))

    word2vec = {}
    with open('datasets/pretrained/wordvecs/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] in uniq_words:
                word2vec[tokens[0]] = [float(x) for x in tokens[1:]]
    print('#word2vec', len(word2vec))

    cat2vec = {}
    for x in categories:
        vec = []
        for w in x.strip().split():
            if w in word2vec:
                vec.append(word2vec[w])
            else:
                print('\t', x, w, 'no exists', uniq_words[w])
        if len(vec) > 0:
            cat2vec[x] = np.mean(vec, 0).tolist()
        else:
            cat2vec[x] = np.zeros(300, ).tolist()
            print(x, 'no exists')

    json.dump(
        cat2vec,
        open(os.path.join(meta_dir, 'cat2glove42b.json'), 'w'),
    )

if __name__ == '__main__':
    main()
    
