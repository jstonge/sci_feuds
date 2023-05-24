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



