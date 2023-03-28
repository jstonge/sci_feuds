from pathlib import Path
from time import sleep

import pandas as pd
import numpy as np
import requests
from jsonlines import jsonlines
from tqdm import tqdm

from creds import client
import re

import umap
import hdbscan

from collections import Counter
import re

import pandas as pd

from creds import client
from tqdm import tqdm

import sys

sys.path.append("../s2orc_helpers/s2orc_helpers")

from s2orc_helpers import parse_res, parse_bibref

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_citing_paperids(typeId):
    if typeId == 'corpusId':
        dat = {}
        with jsonlines.open("broido_scale-free_2019_citations.json") as f:
            for obj in f:
                for paper in obj['data']:
                    if paper['citingPaper']['year'] is not None:
                        pid = paper['citingPaper'][typeId]
                        yr = paper['citingPaper']['year']
                        dat.update({pid: yr})
        return dat  


# ------------------ Grab scale free networks is rare debate ----------------- #

# from bson import json_util
# import json

# corpus = read_citing_paperids('corpusId')

# corpusIds = [k for k in corpus]
# years = [v for v in corpus.values()]

# db = client['papersDB']

# out = [db.s2orc.find_one({'corpusid': cid}) for cid in corpusIds]
# out = [_ for _ in out if _ is not None]

# with open('broido_debate.jsonl', 'w') as fout:
#     for paper in out:
#         json.dump(json.loads(json_util.dumps(paper)), fout)
#         fout.write('\n')


# ----------------------------- Parsing s2orc format ---------------------------- #


# def get_start_headers(paper, annot):
#     headers = paper['content']['annotations'][annot]
#     if headers:
#         return [int(_) for _ in re.findall('(?<=start\":)\d+', headers)]

# def grab_citations(paper, start, end):
#     end_str = re.escape(paper['content']['text'][start:end])
#     return re.findall(rf'([^.]*?{end_str}.+?\.)', paper['content']['text'])


# def get_bibid_source(paper, source):
#     bibentry = paper['content']['annotations']['bibentry']
#     re_pat_split = '({"attributes":{.+?},"end":\d+,"start":\d+})'
#     bibentry = [re.split(",", _) for _ in re.findall(re_pat_split, bibentry)]
    
#     matched_id_re = '(?<=\"matched_paper_id\":)\d+'
#     bib_id_re = '(?<=\"id\":\")[a-z]\d+'
    
#     matched_paper_ids = {
#         int(re.findall(matched_id_re, ' '.join(entry))[0]): re.findall(bib_id_re, ' '.join(entry))[0]
#         for entry in bibentry 
#         if re.search('matched_paper_id', ' '.join(entry))
#     }
#     return matched_paper_ids.get(source)


# def parse_res(paper, annot, source=None):
#     """
#     annotation types:
#     ================
#     - type1: paragraph
#     - type2: figureref, sectionheader -> sections, tableref
#     - type3: 'authors'

#     Desc
#     ====
#     bibauthor: Authors in the bibliography
#     bibentry: Cited papers in the bibliography
#     bibref: Citations in the papers

#      ('abstract', 'author', 'authoraffiliation', 'authorfirstname', 
#       'authorlastname', 'bibauthor', 'bibauthorfirstname', 'bibauthorlastname', 
#       'bibentry', 'bibref', 'bibtitle', 'bibvenue', 'figure', 
#       'formula',  'publisher', 'table', 'title', 'venue')
#     """
#     # subset[0]['content']['annotations'].keys()
#     # paper, annot = papers_broido[0], 'figurecaption'
    
#     paragraphs = paper['content']['annotations']['paragraph']
#     if annot in ['sectionheader', 'figureref', 'tableref', 'figurecaption']:
#         headers = paper['content']['annotations'][annot]
#         paper['content']['text'][103213:103548]
#         if headers is not None:
#             corpusIds = []
#             sections = []
#             content = []
#             # headers = paper['content']['annotations']['tableref']
#             start_headers = [int(_) for _ in re.findall('(?<=start\":)\d+', headers)]
#             tot_sections = len(start_headers)-1
#             if paragraphs is not None:
#                 section = 1
#                 for start, end in zip(start_headers[:-1], start_headers[1:]):
#                     text = paper['content']['text'][start:(end-1)]
#                     corpusIds.append(paper['corpusid'])
#                     sections.append(f'{section}/{tot_sections}')
#                     content.append(text)
#                     section += 1
    
#             return pd.DataFrame({'corpusid': corpusIds, 'sections': sections, 'text': content})
    

#     elif annot in 'bibref':
        
#         return out

# def parse_bibref(paper, source):
#     # paper = papers_broido[0]
#     rel_bibid = get_bibid_source(paper, source)
#     # paper['externalids']
#     bibentry = paper['content']['annotations']['bibref']
#     re_pat_split = '({"attributes":{.+?},"end":\d+,"start":\d+})'
#     bibentry = [re.split(",", _) for _ in re.findall(re_pat_split, bibentry)]

#     out = []
#     for entry in bibentry:
#         # entry = bibentry[0]
#         entries = {}
#         matched_id = re.findall('(?<=\"ref_id\":\")[a-z]\d+' , ', '.join(entry))[0]
#         if matched_id == rel_bibid:
#             for field in entry:
            
#                 k = re.findall("\w+", re.split(":", field)[0])[0]
#                 v = re.findall("\w+", re.split(":", field)[1])[0]
                
#                 if v.isdigit():
#                     v = int(v)

#                 entries.update({k:v})
#             out.append(entries)

    
#     return [grab_citations(paper, ref['start'], ref['end']) for ref in out]

papers_broido = []
with jsonlines.open("broido_debate.jsonl") as f:
    for obj in f:
        papers_broido.append(obj)


parse_res(papers_broido[17], "figurecaption")

source = 24825063
parse_bibref(papers_broido[17], source)

all_cits = []
for paper in papers_broido:
    try:
        parsed_paper = parse_bibref(paper, source)
        all_cits.append(parsed_paper)
    except:
        pass

all_cits = pd.concat(all_cits, axis=0).reset_index(drop=True)

all_cits.citations[3]




excerpt = "CEU’s academic independence, modeled on its US peers, has angered the government, which portrays it as a hotbed of liberal thinking. It is hard to point to any event or action by the university that triggered this crisis. The official reason offered by the government remains puzzling. It argued that U.S.-based degrees offered by CEU are a comparativeadvantage, unmatched by local institutions. In its view,the new law creates an even playing field. This reasoning fooled no one. The law is widely seen as an attemptto gain electoral advantage by picking a fight with theuniversity’s founder, the Hungarian-born U.S. philanthropist George Soros, whose long-standing advocacy for open societies and migrants is at odds with the isolationist stand pursued by Prime Minister Viktor Orbán. The law’s political nature is made manifest in the impossible,and potentially unconstitutional, conditions it imposes. It requires CEU to open a campus in New York State,where it is accredited, by October 2017, which is a practical impossibility. It also requires the university to be regulated by an agreement between Hungary and the U.S.federal government—ignoring the fact that education inthe United States is under the jurisdiction of individualstates. Unable to meet these requirements, CEU will loseits ability to admit new students next spring."
excerpt2 = "The heartwarming response to Lex-CEU reaffirms the power of this interconnectedness. Most academic leaders in Hungary, at great professional and personal risk, have spoken up in support of CEU, and the law prompted large street demonstrations in Budapest."
excerpt3 = "Despite the law’s apparent finality, the battle is just beginning. The university’s president has vowed that research and scholarship will continue. The European Parliament has opened an investigation into the law’s legality and harmony with European Union laws. Within Hungary, the Supreme Court has been asked to rule on the law’s constitutionality, although independence of the courts has been questionable. None of these efforts are likely to conclude by the fatal October deadline, which means that only coordinated and meaningful US and European political pressure, at the highest level, can restore CEU’s ability to enroll its next cohort of students."

re.split("\\. ", excerpt3)

# ----------------------------- Get doc embeddings from specter  ---------------------------- #




def specter_via_api(paperId: str, fout: Path) -> None:
    header = {"x-api-key": "8mH99xWXoi60vMDfkSJtb6zVhSLiSgNP8ewg3nlZ"}
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}"
    fields_papers = "externalIds,isOpenAccess,publicationTypes,publicationDate,journal,embedding,tldr,title,abstract,venue,authors,year,referenceCount,citationCount,influentialCitationCount,openAccessPdf,corpusId,fieldsOfStudy,s2FieldsOfStudy"
    url = f"{base_url}?fields={fields_papers}"

    r = requests.get(url, headers=header)

    if r.status_code == 200:
        with jsonlines.open(fout, "a") as writer:
            writer.write(r.json())
    else:
        return r.status_code

def get_embeddings():
    paperIds = read_citing_paperids('paperId')
    
    outname = Path("specter_embeddings_doc.json")

    for paperid in tqdm(paperIds):
        already_done = []
        if outname.exists():
            with jsonlines.open(outname) as f:
                for line in f:
                    already_done.append(line)
            
            already_done = set([paper["paperId"] for paper in already_done])
            paperIds = set(paperIds) - already_done

        if len(paperIds) > 0:
            counter = 0
            specter_via_api(paperid, outname)
            counter += 1
            if counter == 100:
                sleep(1)
                counter = 0

get_embeddings()

metadata = pd.read_json("specter_embeddings_doc.json", lines=True)
metadata['field_1'] = metadata.s2FieldsOfStudy.map(lambda x: x[0]['category'] if x else None)
doc_embeddings = np.matrix(metadata.embedding.map(lambda x: x['vector']))
umap_args = {'n_neighbors': 15, 'n_components': 2, 'metric': 'cosine'}
hdbscan_args = {'min_cluster_size': 15, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}
doc_ids = metadata.paperId.tolist()
titles = metadata.title.tolist()

umap_model = umap.UMAP(**umap_args).fit(doc_embeddings)
cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)


df = pd.DataFrame({
    'topic': cluster.labels_,
    'paperId': metadata['paperId'],
    'corpusId': metadata['corpusId'],
    'authors': metadata['authors'].map(lambda x: ', '.join([i['name'] for i in x])),
    'field': metadata['field_1'],
    'title': metadata['title'],
    'year' : metadata['year'],
    'venue' : metadata['venue'],
    'abstract': metadata['abstract'],
    'citationCount' : metadata['citationCount']
    })

df['x'] = umap_model.embedding_[:,0]
df['y'] = umap_model.embedding_[:,1]

df.to_parquet(f"broido_embedding_doc.parquet")

