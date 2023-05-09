import json
import re

from io import BytesIO
from pathlib import Path
from textwrap import wrap
from time import sleep
from tqdm import tqdm


import bibtexparser
import pandas as pd
from bson import json_util
from jsonlines import jsonlines

from creds import client

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_bibtex(fname):
    with open(fname) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database

def get_id_from_url(x):
    return re.sub("https://www.semanticscholar.org/paper/", "", x)

def map2int_5(x):
    if x <= -0.6:
        return '1'
    elif x <= -0.2:
        return '2'
    elif x <= 0.2:
        return '3'
    elif x <= 0.6:
        return '4'
    else:
        return '5'

def map2int_3(x):
    if x > 0.75:
        return 'positive'
    elif x > -0.25:
        return 'neutral'
    else:
        return 'negative'

def map2int_9(x):
    if x == -1.:
        return 1
    elif x <= -0.7:
        return 2
    elif x <= -0.4:
        return 3
    elif x <= -0.1:
        return 4
    elif x <= 0.1:
        return 5
    elif x <= 0.4:
        return 6.
    elif x <= 0.7:
        return 7
    elif x < 1.:
        return 8
    elif x == 1.:
        return 9


def read_meta_papers(pid):
    meta_papers=[]
    with open(f"../feuds/{pid}.jsonl") as f:
        for line in f:
            meta_papers.append(json.loads(line))

    meta_papers = meta_papers[0] if len(meta_papers) == 1 else meta_papers
    return meta_papers

def get_seeds_fight_papers():
    
    bib_database = read_bibtex('../data/scientific_feuds.bib')

    assert all([p.get('keywords') is not None for p in bib_database.entries]), 'all papers must be labeled with a feud'

    # need to be connected to uvm server to work
    db = client['papersDB']
    
    # we want to find the semantic scholar paperIds + full metadata about paper
    urls = ["https://www.semanticscholar.org/paper/"+re.sub("paper\\\\_id\.", "", _['annote']) 
            for _ in bib_database.entries if _.get('annote')]

    all_pids = [db.papers.find_one({'url': u}) for u in urls]

    for i,pid in enumerate(all_pids):
        print(i)
        pid['zot_id'] = bib_database.entries[i]['ID']
        pid['zot_keywords'] = bib_database.entries[i]['keywords']

    json_dump  = json_util.dumps(all_pids)

    with jsonlines.open('../data/seeds_fights_papers.jsonl', mode='w') as writer:
        writer.write(json.loads(json_dump))



# ----------------------------- Add citation tags ---------------------------- #


def substite_cite_tag(text, x):
    """
    text: string
    x: cite_span of the form
      {'start': 187, 'end': 191, 'text': '[20]', 'ref_id': 'BIBREF19'}
    """
    # assert x.get("start") and x.get("end") and x.get('text')
    return text[:x['start']] + "<cite>" + x['text'] + "</cite>" + text[x['end']+1:]


def ref_id2name_lookup(x, ref_id_authors):
    if x is not None:
        return ref_id_authors[re.sub("IBREF", "", x).lower()]


def concat_name(x):
    if x is None:
        return None
    if isinstance(x['middle'], list):
        return x['first'] + " " + x['last']
    else:
        return x['first'] + " " + x['middle'] + " " + x['last']


def add_cit_tags(input_dir, df_meta, output_dir):
    file2do = list(input_dir.glob("*json"))
    all_cits = []
    # df_meta = pd.read_csv(OUTPUT_DIR/"groupSel_feud.csv", usecols=['citationCounts', 'ID', 'year', 'author'])
    for article in tqdm(file2do):
        # article = file2do[2]
        # print(article)
        fname = re.sub("\.json", "", str(article).split("/")[-1])
        
        with open(article) as f:
            dat = json.load(f)
            
            # Get text from s2orc metadata
            texts = [_['text'] for _ in dat['pdf_parse']['body_text']]
            cite_spans = [_['cite_spans'] for _ in dat['pdf_parse']['body_text']]
            ref_id_authors = {_['ref_id']: _['authors'] for _ in dat['pdf_parse']['bib_entries'].values()}
            
            [re.sub("group selection", "<cite>group selection</cite>", text) 
             for text in texts if bool(re.search("group selection", text))]

            all_texts_cite = []
            all_spans_cite = []
            for text, cite_span in zip(texts, cite_spans):
                # print(cite_span, i)
                if len(cite_span) > 0:
                    all_texts_cite.append([substite_cite_tag(text, span) for span in cite_span])
                    all_spans_cite.append([ref_id2name_lookup(span['ref_id'], ref_id_authors) for span in cite_span])
                    
                else:
                    all_texts_cite.append([])
                    all_spans_cite.append([])
        
        all_cits.append((fname, all_texts_cite, all_spans_cite))

    df_cit = pd.DataFrame(all_cits, columns=['article', 'parsed_doc', 'cite_spans'])
    df_cit = df_cit.merge(df_meta, how='left', left_on="article", right_on="ID")
    
    # explode document & cite_spans
    df_cit_long = df_cit.explode(["parsed_doc", 'cite_spans'])
    df_cit_long = df_cit_long[df_cit_long.parsed_doc.map(lambda x: len(x) if isinstance(x, list) else 0) > 0]
    df_cit_long = df_cit_long.reset_index().rename(columns={'index': 'did'})

    # explode paragraphs
    df_cit_long = df_cit_long.explode(['parsed_doc', 'cite_spans']).reset_index().rename(columns={'index': 'sid'})

    # explode authors
    df_cit_long = df_cit_long.explode('cite_spans').reset_index().rename(columns={'index': 'aid'})  
    df_cit_long['cite_spans'] = df_cit_long.cite_spans.map(lambda x: concat_name(x) if isinstance(x, dict) else None)

    # remove any NA
    df_cit_long = df_cit_long[~df_cit_long.cite_spans.isna()].reset_index(drop=True)

    df_cit_long['year'] = pd.to_datetime(df_cit_long['year'], format="%Y-%m-%d")


    df_cit_long.to_parquet(output_dir / "groupSel_feud_with_tag.parqet", index=False)

