import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from tqdm import tqdm

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'

from helpers import flatten


def update_filetodo():
    done_files = list(SPACY_DIR.joinpath('sent').glob("*pqt"))
    file2do = list(GROBID_DIR.glob("*json"))
    
    if len(done_files) == 0:
        return list(GROBID_DIR.glob("*json"))
    
    done_fname = set([re.sub("(_sent|\.pqt)", "", str(file).split("/")[-1]) for file in done_files])
    file2do  = [file for file in file2do if re.sub("(_sent|\.json)", "", str(file).split("/")[-1]) not in done_fname]
    print(f"Remaining {len(file2do)} files")
    return file2do



#! TODO: finish that
def spacy_parse_feud():    
    file2do = update_filetodo()
    
    if len(file2do) > 0:
        nlp = spacy.load("en_core_web_trf")
        for article in tqdm(file2do):
            # article=file2do[0]
            all_docs = []
            all_toks = []
            fname = re.sub("\.json", "", str(article).split("/")[-1])
            print(f"doing {fname}")
            
            with open(article) as f:
                dat = json.load(f)
                
                # Get text from s2orc metadata
                texts = [_['text'] for _ in dat['pdf_parse']['body_text']]

                # Parsed into spacy documents
                docs = list(nlp.pipe(texts))
                
                # doc level
                for i, doc in enumerate(docs):
                    out = []
                    
                    # sentence level
                    for j, sent in enumerate(doc.sents):
                        all_docs.append((i, j,sent.text, fname))

                        # token level
                        for token in sent:
                            out.append((i, j, token.text, token.lemma_, 
                                        token.pos_, token.tag_, token.dep_,
                                        token.shape_, token.is_alpha, token.is_stop, fname))
                    
                    all_toks.append(out)
                
            df_sent = pd.DataFrame(all_docs, columns=["did", "sid", "sentence", "article"])
            df_sent['uniq_id'] = df_sent.article+"_"+df_sent.did.astype(str)+"_"+df_sent.sid.astype(str)
            df_sent.to_parquet(OUTPUT_DIR / "spacy_group_selection_grobid" / f"{fname}_sent.pqt", index=False)

            all_toks_comb = flatten(all_toks)
            df_toks = pd.DataFrame(all_toks_comb, columns=["did", "sid", "text", "lemma", "pos", "tag", "dep", "shape", "is_alpha", "is_stop", "article"])
            df_toks['uniq_id'] = df_toks.article+"_"+df_toks.did.astype(str)+"_"+df_toks.sid.astype(str)
            df_toks.to_parquet(OUTPUT_DIR / "spacy_group_selection_grobid" / f"{fname}_toks.pqt", index=False)
    else:
        print("done")

spacy_parse_feud()