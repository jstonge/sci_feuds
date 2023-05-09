
import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'

from helpers import flatten

# ----------------- extracting labelled data in a tidy format ---------------- #



df=pd.read_csv("../output/spacy_group_selection_grobid/coref_par_w_ent_1960_1970.csv")
par_id = df['uniq_id'].map(lambda x: x.split("', ")[0].split("_")[-2])
article = df['article']
df["article_par_id"] = article+'_'+par_id

with open("../output/groupSel_annotated_labelStudio_2023-04-26.json") as f:
    dat = json.load(f)

def _converter(df_res, source_id, target_id):   
    source = df_res[df_res['id'] == source_id].value.iloc[0]
    type = source['labels'][0]
    source = source['text']
    target = df_res[df_res['id'] == target_id].value.iloc[0]
    sentiment = int(target['labels'][0])
    target = target['text']
    return [source, target, sentiment, type]

def convert_all():
    """convert all paragraph from label-studio to tidy format"""
    dfs = []
    for d in dat:
        print(d['data']['article'])
        res = d['annotations'][0]['result']
        df_res=pd.DataFrame(res)
        source_ids, target_ids = zip(*[(_['from_id'], _['to_id']) for _ in res 
                                    if _.get('from_id') and _.get('to_id')])
        list_relations = [_converter(df_res, source_id, target_id) for source_id, target_id in zip(source_ids, target_ids)]
        df=pd.DataFrame(list_relations, columns=['target', 'text', 'sentiment', 'type'])
        par_id = d['data']['uniq_id'].split("', ")[0].split("_")[-2]
        article = d['data']['article']
        df["article_par_id"] = article+'_'+par_id
        dfs.append(df)

    return pd.concat(dfs, axis=0)

df_annot = convert_all()

sub_df = df_annot[df_annot.article_par_id == 'odonald_general_1967_20']
completion = [tuple(_) for _ in sub_df[['text', 'sentiment', 'target']].to_numpy()]

coref_text = df[df.article_par_id == 'odonald_general_1967_20'].coref_sent.tolist()[0]
prompt = "The main clauses and associated entities in this sentence are: {coref_text}"

{"prompt": prompt, "completion": json.dumps(completion)}


