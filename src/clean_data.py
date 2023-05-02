import json
import re
from pathlib import Path

import pandas as pd
import spacy
import tqdm
from tqdm import tqdm

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'


def flatten(l):
    return [item for sublist in l for item in sublist]

def update_filetodo():
    done_files = list(SPACY_DIR.joinpath('sent').glob("*pqt"))
    file2do = list(GROBID_DIR.glob("*json"))
    
    if len(done_files) == 0:
        return list(GROBID_DIR.glob("*json"))
    
    done_fname = set([re.sub("(_sent|\.pqt)", "", str(file).split("/")[-1]) for file in done_files])
    file2do  = [file for file in file2do if re.sub("(_sent|\.json)", "", str(file).split("/")[-1]) not in done_fname]
    print(f"Remaining {len(file2do)} files")
    return file2do



# ------------- cleaning data - get spacy representation of group selection feud ------------- #


def spacy_parse_feud():    
    file2do = update_filetodo()
    
    if len(file2do) > 0:
        nlp = spacy.load("en_core_web_trf")
        for article in tqdm(file2do):
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


# ------------ cleaning data - keeping only sentence with entities ----------- #


def filter_out_sentence_wo_ent(start=1960, end=2023, by_paragraph=False):
    dfs = []
    df_meta=pd.read_csv(OUTPUT_DIR / "groupSel_feud.csv")
    
    for article in tqdm(list(SPACY_DIR.joinpath('sent').glob("*pqt"))):
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


def flatten(l):
    return [item for sublist in l for item in sublist]


import spacy
from fastcoref import spacy_component

df=pd.read_csv("par_w_ent_1960_1970.csv")
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
out=list(nlp.pipe(df.sentence.tolist(), component_cfg={"fastcoref": {'resolve_text': True}}))
df['coref_sent'] = flatten(out)
df['coref_sent'] = df['coref_sent'].map(lambda x: x._.resolved_text)
