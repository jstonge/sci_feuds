import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import spacy
from tqdm import tqdm

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'

from helpers import flatten


# ------------ cleaning data - keeping only sentence with entities ----------- #


def filter_out_sentence_wo_ent(start=1960, end=2023, by_paragraph=False):
    dfs = []
    df_meta=pd.read_csv(OUTPUT_DIR / "groupSel_feud.csv")
    list_fnames = list(SPACY_DIR.joinpath('sent').glob("*pqt"))
    for article in tqdm(list_fnames):
        article = list_fnames[0]
        # article = '../output/spacy_group_selection_grobid/sent/nelson_clutch_1966_sent.pqt'
        # article=list(SPACY_DIR.joinpath('sent').glob("*pqt"))[0]
        fname = re.sub("_sent.pqt", "", str(article).split("/")[-1])
        yr = int(re.findall("\d{4}", fname.split("_")[-1])[0])
        if yr >= start and yr < end:
            # print(article)
            df_sent=pd.read_parquet(SPACY_DIR / 'sent' / f"{fname}_sent.pqt")
            df_toks=pd.read_parquet(SPACY_DIR / "toks" / f"{fname}_toks.pqt")
            
            id2keep = set(df_toks[df_toks.pos.isin(["PROPN","SYM"])]
                            .reset_index(drop=True)
                            .loc[:, ["text","tag", "dep", "pos", "uniq_id"]]
                            .uniq_id)
            
            if by_paragraph:
                if len(df_sent) == 0:
                    dfs.append(df_sent[['sentence', 'uniq_id', 'article']].assign(citationCounts = ''))
                else:
                    df_par = df_sent[['sentence', 'did', 'article', 'uniq_id']].groupby(['did']).agg({
                        'sentence': lambda x: ' '.join(x), 
                        'uniq_id': lambda x: x.to_list(),
                        'article': lambda x: x.to_list()[0],
                        })
                
                    df_par_f = df_par[df_par.uniq_id.map(lambda x: len(set(x) & id2keep) > 0)].reset_index(drop=True)
                    df_par_f = df_par_f.merge(df_meta[['ID', 'citationCounts']], left_on="article", right_on="ID", how="left").drop('ID',axis=1)
                    dfs.append(df_par_f)
            else:
                df_sent_f = df_sent[df_sent.uniq_id.isin(id2keep)]
                df_sent_f = df_sent_f.merge(df_meta[['ID', 'citationCounts']], left_on="article", right_on="ID", how="left").drop('ID',axis=1)
                dfs.append(df_sent_f)

    df =  pd.concat(dfs, axis=0)
    df['wc'] = df.sentence.str.split(" ").map(len)
    
    return df.sample(len(df))

yr1, yr2 = 1960, 2023
df_sent_w_ent2 = filter_out_sentence_wo_ent(start=yr1, end=yr2, by_paragraph=True)
df_sent_w_ent2.to_csv(SPACY_DIR / f"par_w_ent_{yr1}_{yr2}.csv", index=False)



# -------------------------- coreference resolution -------------------------- #



import spacy
from fastcoref import spacy_component

df=pd.read_csv("par_w_ent_1960_1970.csv")
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
out=list(nlp.pipe(df.sentence.tolist(), component_cfg={"fastcoref": {'resolve_text': True}}))
df['coref_sent'] = flatten(out)
df['coref_sent'] = df['coref_sent'].map(lambda x: x._.resolved_text)
