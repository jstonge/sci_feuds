
import re

import bibtexparser
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from creds import client


def read_bibtex(fname):
    with open(fname) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database

def get_df_pattern(x, db):   
    re_pattern = re.compile(x, re.IGNORECASE)
    res = list(db.abstracts.find({"abstract": {"$regex": re_pattern}}))
    return pd.DataFrame(res)

def augment_meta_df(df):
    meta = []
    for cid in tqdm(df.corpusid):
        meta.append(db.papers.find_one({"corpusid" : cid}))

    meta_gs_df = pd.DataFrame(meta)
    df = df.merge(meta_gs_df, on='corpusid', how='left') 
    df['year'] = pd.to_datetime(df.year, format="%Y")
    return df

# load db
db = client['papersDB']

bibdatabase = read_bibtex('../data/group_selection.bib')

paper_ids = ["https://www.semanticscholar.org/paper/"+re.sub('paper\\\_id\.', '', e['annote']) for e in bibdatabase.entries]

meta_group_sel_zot = [db.papers.find_one({'url': pid}) for pid in paper_ids]
cids_group_sel_zot = set([_['corpusid'] for _ in meta_group_sel_zot if _])

abstract_group_sel_zot = [db.abstracts.find_one({'corpusid': cid}) for cid in cids_group_sel_zot]

papers_w_group_sel = [_ for _ in abstract_group_sel_zot if _ and bool(re.search("group selection", _['abstract']))]
cid_papers_w_group_sel = set([_['corpusid'] for _ in papers_w_group_sel])

df_gs = get_df_pattern('group selection', db)
df_mls = get_df_pattern('multilevel selection', db)

cid_db_gs = set(df_gs.corpusid.tolist())

# len(cid_papers_w_group_sel) / len([_ for _ in abstract_group_sel_zot if _]) * 100
# len(cid_papers_w_group_sel) / len(cid_db_gs) * 100
# len(cids_group_sel_zot & cid_db_gs)
# print(f"Proportion of papers in zot we have in our database: {len(meta_group_sel_zot) / len(paper_ids)}")


df_gs = augment_meta_df(df_gs)
df_mls = augment_meta_df(df_mls)

fig, ax = plt.subplots(1,1, figsize=(8,5))
df_mls.value_counts('year').rename('multi-level selection').plot(ax=ax, legend='MLS')
df_gs.value_counts('year').rename('group selection').plot(ax=ax, legend='group selection')




# temporal analysis based on Liu & Zhu 2023

# hard discipline: aerospace, automation and control, software engineering, transportation
# soft discpline: comm, ling, pol sci, socio

foo=list(db.papers.find({"year": 2000, "citationcount": { "$gte": 0 }, "s2fieldsofstudy": { "$exists": True } }))
foo=list(db.papers.find({"year": 2000 })).limit(500)


# def normalized_citation_count(target_p, background_yr):
#     """papers object"""
#     if target_p.get('citationcount'):
#         raw_citation_count = target_p['citationcount']
#         tot_articles_this_yr = 
#         sum_citation_count_received_by_all_article_yr = 

# def specter_via_api(paperId: str, fout: Path) -> None:
#     header = {"x-api-key": "8mH99xWXoi60vMDfkSJtb6zVhSLiSgNP8ewg3nlZ"}
#     base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}"
#     fields_papers = "externalIds,isOpenAccess,publicationTypes,publicationDate,journal,embedding,tldr,title,abstract,venue,authors,year,referenceCount,citationCount,influentialCitationCount,openAccessPdf,corpusId,fieldsOfStudy,s2FieldsOfStudy"
#     url = f"{base_url}?fields={fields_papers}"

#     r = requests.get(url, headers=header)

#     if r.status_code == 200:
#         with jsonlines.open(fout, "a") as writer:
#             writer.write(r.json())
#     else:
#         return r.status_code

# def get_embeddings():
#     paperIds = read_citing_paperids('paperId')
    
#     outname = Path("specter_embeddings_doc.json")

#     for paperid in tqdm(paperIds):
#         already_done = []
#         if outname.exists():
#             with jsonlines.open(outname) as f:
#                 for line in f:
#                     already_done.append(line)
            
#             already_done = set([paper["paperId"] for paper in already_done])
#             paperIds = set(paperIds) - already_done

#         if len(paperIds) > 0:
#             counter = 0
#             specter_via_api(paperid, outname)
#             counter += 1
#             if counter == 100:
#                 sleep(1)
#                 counter = 0

# get_embeddings()

# metadata = pd.read_json("specter_embeddings_doc.json", lines=True)
# metadata['field_1'] = metadata.s2FieldsOfStudy.map(lambda x: x[0]['category'] if x else None)
# doc_embeddings = np.matrix(metadata.embedding.map(lambda x: x['vector']))
# umap_args = {'n_neighbors': 15, 'n_components': 2, 'metric': 'cosine'}
# hdbscan_args = {'min_cluster_size': 15, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}
# doc_ids = metadata.paperId.tolist()
# titles = metadata.title.tolist()

# umap_model = umap.UMAP(**umap_args).fit(doc_embeddings)
# cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)


# df = pd.DataFrame({
#     'topic': cluster.labels_,
#     'paperId': metadata['paperId'],
#     'corpusId': metadata['corpusId'],
#     'authors': metadata['authors'].map(lambda x: ', '.join([i['name'] for i in x])),
#     'field': metadata['field_1'],
#     'title': metadata['title'],
#     'year' : metadata['year'],
#     'venue' : metadata['venue'],
#     'abstract': metadata['abstract'],
#     'citationCount' : metadata['citationCount']
#     })

# df['x'] = umap_model.embedding_[:,0]
# df['y'] = umap_model.embedding_[:,1]

# df.to_parquet(f"broido_embedding_doc.parquet")

