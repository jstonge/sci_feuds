import argparse
from pathlib import Path
import pathlib
import re

import numpy as np
import pandas as pd

from top2vec import Top2Vec


def main():


    doc_embeddings = np.concatenate([np.load(mat) for mat in fnames], axis=0)

    colnames = ['paperId', 'title', 'abstract', 'year', 'venue', 'citationCount', 'field']
    metadata = pd.read_csv(meta_embedding_out, sep="\t", names=colnames)

    assert len(metadata) == doc_embeddings.shape[0], "Number of metadata should be the same than number of cols in embeddings"

    undup_idx = np.where(~metadata.paperId.duplicated())[0]
    print(f"{len(metadata) - len(undup_idx)} duplicated indices")
    cols = np.array(range(doc_embeddings.shape[1]))

    doc_embeddings = doc_embeddings[np.ix_(undup_idx, cols)]
    metadata = metadata.loc[undup_idx,:].reset_index(drop=True)

    # N_REP = 20_000
    idx_subset = metadata.groupby(['field'], as_index=False, group_keys=False
            ).apply(lambda x: x.sample(min(args.N_REP, len(x)))).index.to_numpy()

    doc_embeddings_subset = doc_embeddings[np.ix_(idx_subset,cols)]
    doc_ids = metadata.loc[idx_subset,'paperId'].tolist()
    abstracts = metadata.loc[idx_subset,'abstract'].tolist()
    umap_args = {'n_neighbors': 15, 'n_components': args.N_comp, 'metric': 'cosine'}

    model = Top2Vec(doc_embeddings_subset, doc_ids, abstracts, umap_args)

    df = pd.DataFrame({
        'topic': model.doc_top,
        'paperId': metadata.loc[idx_subset,'paperId'],
        'field': metadata.loc[idx_subset,'field'],
        'title': metadata.loc[idx_subset,'title'],
        'year' : metadata.loc[idx_subset,'year'],
        'venue' : metadata.loc[idx_subset,'venue'],
        'citationCount' : metadata.loc[idx_subset,'citationCount']
        })

    if args.N_comp == 2:
        df['x'] = model.embedding[:,0]
        df['y'] = model.embedding[:,1]

    df.to_parquet(f"umap_embedding_{args.N_comp}NCOMP.parquet")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path)
    parser.add_argument("--N_REP", type=int)
    parser.add_argument("--N_comp", type=int)
    args = parser.parse_args()

    embedding_out = args.input_dir / "embeddings.npy"
    meta_embedding_out = args.input_dir / 'meta_embeddings.tsv'

    main()