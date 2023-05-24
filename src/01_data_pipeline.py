"""
Description: 
 - We put relevant articles in a zotero collection
 - Given too many text files are missing from S2ORC, we scraped relevant pdf.
 - 
"""
import json
import re
import shutil
import sys
from pathlib import Path
from textwrap import wrap
from tqdm import tqdm

import pandas as pd
import requests
from bson import json_util
from jsonlines import jsonlines

from creds import client

sys.path.append("../../s2orc_helpers/s2orc_helpers")

from s2orc_helpers import parse_bibref

from helpers import get_id_from_url, read_bibtex, substite_cite_tag, concat_name, ref_id2name_lookup

ROOT_DIR = Path("../")
DATA_DIR = ROOT_DIR / 'data'
PDF_DIR = DATA_DIR / 'raw_pdfs'
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'
CCA_DIR = OUTPUT_DIR / 'cca'

# ---------------------------------- helpers --------------------------------- #


def get_citation_graph(paper) -> None:
    # paper = feud_seeds[13]
    paperId = re.sub("https://www.semanticscholar.org/paper/", "", paper['url'])
    header = {"x-api-key": "8mH99xWXoi60vMDfkSJtb6zVhSLiSgNP8ewg3nlZ"}
    
    fields_papers = "contexts,intents,isInfluential,externalIds,isOpenAccess,publicationTypes,publicationDate,journal,title,abstract,venue,authors,year,referenceCount,citationCount,influentialCitationCount,openAccessPdf,corpusId,fieldsOfStudy,s2FieldsOfStudy"

    offset = 0
    limit = paper['citationcount'] if paper['citationcount'] < 1000 else 1000
    total_citations = []

    while True:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}/citations?fields={fields_papers}&offset={offset}&limit={limit}"
        r = requests.get(url, headers=header)
        if r.status_code == 200:
            print(f"Doing {offset}-{offset+limit}")
            citations = r.json()['data']
            total_citations += citations
            if len(citations) < limit or len(total_citations) >= 9999:
                break
            else:
                offset += limit if offset < 9000 else 999

        else:
            print(f'Request failed with status code {r.status_code}')
            break


    with jsonlines.open(f"../feuds/{paperId}.jsonl", "w") as writer:
        writer.write(total_citations)

def read_citing_paperids(fname):
    with jsonlines.open(fname) as f:
        dat = f.read()
    
    cids = [_['citingPaper']['corpusId'] for _ in dat]
    yrs = [_['citingPaper']['year'] for _ in dat]

    return cids, yrs

def get_papers():
    ROOT_DIR = Path("..") 
    files = ROOT_DIR.joinpath("data").glob("*jsonl")
    files_done = set([re.sub(".jsonl", "", str(_).split("/")[-1]) for _ in files])
    files2todo = [_ for _ in ROOT_DIR.joinpath("feuds").glob("*jsonl")
                  if re.sub(".jsonl", "", str(_).split("/")[-1]) not in files_done]

    for file in files2todo:
        # file=files2todo[0]
        corpusIds, years = read_citing_paperids(file)
        
        db = client['papersDB']
        
        out = [db.s2orc.find_one({'corpusid': cid}) for cid in corpusIds]
        out = [_ for _ in out if _ is not None]

        outname = ROOT_DIR / 'data' / str(file).split("/")[-1]
        with open(outname, 'w') as fout:
            for paper in out:
                json.dump(json.loads(json_util.dumps(paper)), fout)
                fout.write('\n')

def id2shortname():   

    def get_short_name(p):
        return p['authors'][0]['name'].split(" ")[-1] + str(p['year'])

    return { get_id_from_url(p['url']) : get_short_name(p) for p in feud_seeds }

def get_bibid_source(citing_paper, cited_paper):
    out = None
    cited_corpusid = cited_paper['corpusid']
    cited_doi = cited_paper['externalids'].get('DOI')
    bibentry = citing_paper['content']['annotations'].get('bibentry')
    if bibentry:
        bibentry=json.loads(bibentry)
        for bib in bibentry:
            if 'matched_paper_id' in list(bib['attributes']):
                if bib['attributes']['matched_paper_id'] == cited_corpusid:
                    out = bib['attributes']['id']
                
            elif 'doi' in list(bib['attributes']):
                if bib['attributes']['doi'] == cited_doi:
                        out = bib['attributes']['id']
        return out

def parse_bibref(citing_paper, meta_citing_paper, cited_paper):
    # meta_citing_paper=meta_papers
    # [i for i,p in enumerate(papers) if p['corpusid'] == 255367277]
    # citing_paper=papers[1]'
    print(citing_paper['corpusid'])
    
    citing_paper_meta = [_ for _ in meta_citing_paper 
                         if _['citingPaper']['corpusId'] == citing_paper['corpusid']][0]

    # Bib id of the cited paper in the citing paper
    rel_bibid = get_bibid_source(citing_paper, cited_paper)
    bibref = citing_paper['content']['annotations']['bibref']
    pars = citing_paper['content']['annotations']['paragraph']

    if rel_bibid and bibref and pars:
        
        bibentry = [bib for bib in json.loads(bibref) if bib.get('attributes')]
        paragraphs = json.loads(pars)
        
        citation_content = []
        start_end = [(int(bib['start']), int(bib['end'])) for bib in bibentry if bib['attributes']['ref_id'] == rel_bibid]
        for start, end in start_end:
            # print((start, end))
            # start, end = 12269, 12272
            for p in paragraphs:
                # p=paragraphs[0]
                p_start, p_end = int(p['start']), int(p['end'])
                if start > p_start and end < p_end:
                    citation_content.append(citing_paper['content']['text'][p_start:p_end])
                    break 

        out = {'source_corpusid': citing_paper['corpusid'], 'target_corpusid': citing_paper['corpusid'],  'target_bibid': rel_bibid, 'citations': citation_content}
        out.update(citing_paper_meta)
        return out
    else:
        # print(f"{citing_paper['corpusid']} failed")
        pass

def read_meta_papers(pid):
    meta_papers=[]
    with open(f"../feuds/{pid}.jsonl") as f:
        for line in f:
            meta_papers.append(json.loads(line))

    meta_papers = meta_papers[0] if len(meta_papers) == 1 else meta_papers
    return meta_papers

def read_papers(pid):
    papers = []
    with open(f"../data/{pid}.jsonl") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

def print_summary():
    print(f"target paper: {target_paper[0]['title']} by {target_paper[0]['authors'][0]['name']}", end="\n")
    print(f"debate: {target_paper[0]['zot_keywords']}", end="\n")
    print(f"citation count: {target_paper[0]['citationcount']} (we have {len(all_citations)})")


# ----------------------------- checking the raw data ---------------------------- #


bibdatabase = read_bibtex(DATA_DIR / 'group_selection.bib')

# Pdfs duplicated by entry 

local_zotero_storage = "/home/jstonge/Zotero/storage"

dup_pdfs = [e['file'] if e.get('file') else None for e in bibdatabase.entries 
            if e.get('file') 
            and len(re.findall(f"{local_zotero_storage}/\w+", e['file'])) > 1]


# Some problems with the data
#  - Entries without paperIds, or wrong value.
#    >>> {_['ID']: _['annote'] for _ in bibdatabase.entries if len(_['annote']) != 50}
#  - Duplicated titles
#    >>> all_titles = [entry['title'].lower() for entry in bibdatabase.entries]
#    >>> titles_counts = Counter(all_titles)
#    >>> {k:v for k,v in titles_counts.items() if v != 1}
#  - Extracting Citation; 
#    - Books have parsing other than citation counts in extra
#      >>> no_cit_count = [entry for entry in bibdatabase.entries if entry.get('note') and len(entry.get('note')) == 1]


# ------------------------- create metadata dataframe ------------------------ #


def create_meta_df_from_bib():
    ids = [e['ID'] for e in bibdatabase.entries]
    titles = [e['title'] for e in bibdatabase.entries]
    years = [f"{e['year']}-01-01" for e in bibdatabase.entries]
    abstracts = ["\n".join(wrap(e['abstract'], width=200)) if e.get('abstract') else None for e in bibdatabase.entries]
    cit_counts = [int(re.findall("^\d+", entry['note'])[0]) if entry.get('note') and len(re.findall("^\d+", entry['note'])) > 0 else 0  for entry in bibdatabase.entries]
    authors = [e.get('author') for e in bibdatabase.entries]
    pd.DataFrame({'citationCounts': cit_counts, 'ID': ids, 'title': titles, 'year': years, 'abstract': abstracts, 'author': authors})\
      .to_csv(OUTPUT_DIR / "groupSel_feud.csv", index=False)


create_meta_df_from_bib()


# ------------------------------- wrangling pdfs ------------------------------- #


def mv_and_rename_pdfs_in_proj():
    """
    For some entry, we have multiple pdfs. 
    We simply grab the first one at the moment.
    We rename pdfs to be `short_id` from S2ORC
    """
    bib = read_bibtex('../data/group_selection.bib')

    feud = 'group_selection'
    FEUD_DIR = PDF_DIR / feud

    pdf_paths = [Path(re.findall("/home/jstonge/Zotero/storage/\w+", e['file'])[0]) if e.get('file') else None for e in bibdatabase.entries if e.get('file')]
    short_ids = [e['ID'] if e.get('file') else None for e in bib.entries if e.get('file')]
    
    print(f"{round(len(pdf_paths) / len(bib.entries) * 100, 2)}% PDF coverage")

    if FEUD_DIR.is_dir():
        for i, (pdf_path, short_id) in enumerate(zip(pdf_paths, short_ids)):
            pdf = list(pdf_path.glob("*pdf"))
            if len(pdf) >= 1:
                pdf = pdf[0]
                print(i)
                shutil.copy(pdf, FEUD_DIR)
                p = FEUD_DIR /  str(pdf).split("/")[-1]
                target = FEUD_DIR / Path(short_id+".pdf")
                p.rename(target)


mv_and_rename_pdfs_in_proj()


# ------------------------------ GROBID parsing ------------------------------ #

# Now that we have the pdf scrapped and renamed after metdata in S2ORC, 
# we use the parser from [s2orc-doc2json](https://github.com/allenai/s2orc-doc2json).
# We store the output in `output/group_selection_grobid`.

#! for file in group_selection/*pdf; do python process_pdf.py -i $file  -t temp_dir/ -o output_dir/; done


# ----------------------------- add citation tags ---------------------------- #


df_meta = pd.read_csv(OUTPUT_DIR/"groupSel_feud.csv", usecols=['citationCounts', 'ID', 'year', 'author'])

def add_cit_tags():
    file2do = list(GROBID_DIR.glob("*json"))
    all_cits = []
    # df_meta = pd.read_csv(OUTPUT_DIR/"groupSel_feud.csv", usecols=['citationCounts', 'ID', 'year', 'author'])
    for article in tqdm(file2do):
        # article = file2do[0]
        # print(article)
        fname = re.sub("\.json", "", str(article).split("/")[-1])
        
        with open(article) as f:
            dat = json.load(f)
            
            # Get text from s2orc metadata
            texts = [_['text'] for _ in dat['pdf_parse']['body_text']]
            cite_spans = [_['cite_spans'] for _ in dat['pdf_parse']['body_text']]
            ref_id_authors = {_['ref_id']: _['authors'] for _ in dat['pdf_parse']['bib_entries'].values()}
            
            all_texts_cite = []
            all_spans_cite = []
            for text, cite_span in zip(texts, cite_spans):
                # print(cite_span, i)
                # text = texts[4]
                # cite_span = cite_spans[4]
                if len(cite_span) > 0:
                    text_with_tags = [substite_cite_tag(text, span) for span in cite_span]
                    all_texts_cite.append(text_with_tags)
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

    # remove duplicated
    df_cit_long = df_cit_long[~df_cit_long.duplicated(['did', 'sid', 'aid'])]

    df_cit_long.to_parquet(OUTPUT_DIR / "groupSel_feud_with_tag.parquet", index=False)


add_cit_tags(GROBID_DIR, df_meta, OUTPUT_DIR)



# checkpoint observable
def save_author_observable(tidy_df):
    sub_df = tidy_df[tidy_df.cite_spans.map(len) > 5]
    top_15_names = sub_df.value_counts('cite_spans').head(15).reset_index(name='n').cite_spans
    top_50_names = sub_df.value_counts('cite_spans').head(50).reset_index(name='n').cite_spans

    top_15_df = sub_df.loc[sub_df.cite_spans.isin(top_15_names), ['cite_spans', 'year', 'article']]
    top_15_df.to_parquet(OUTPUT_DIR/"cited_authors_top_15.parquet", index=False)
    
    top_50_df = sub_df.loc[sub_df.cite_spans.isin(top_50_names), ['cite_spans', 'year', 'article']]
    top_50_df.to_parquet(OUTPUT_DIR/"cited_authors_top_50.parquet", index=False)

df_cit_long = pd.read_parquet(CCA_DIR / "groupSel_feud_with_tag.parquet")
save_author_observable(df_cit_long)









# --------------------- Examining results at paper level ---------------------------- #


# with jsonlines.open("../data/seeds_fights_papers.jsonl") as f:
#     feud_seeds = f.read()

# all_pids = [get_id_from_url(f['url']) for f in feud_seeds]

# pid='9bec2ca8875bf7179e6b3371b5328a6af5af55de'
# target_paper = [_ for _ in feud_seeds if get_id_from_url(_['url']) == pid]
# meta_papers=read_meta_papers(pid)
# papers=read_papers(pid)
# all_citations = [parse_bibref(p, meta_papers, target_paper[0]) for p in papers]

# print_summary()

# def plot_all():
#     dfs = []
    
#     for pid in all_pids:
#         print(pid)
#         target_paper = [_ for _ in feud_seeds if get_id_from_url(_['url']) == pid]
#         meta_papers=read_meta_papers(pid)
#         papers=read_papers(pid)
        
#         all_citations = [parse_bibref(p, meta_papers, target_paper[0]) for p in papers]
        
#         # [f"Paper {i} cites {len(cit['citations'])}" for i, cit in enumerate(all_citations) if cit is not None and cit.get('citations')]

#         count_cit = Counter([len(c['citations']) for c in all_citations if c is not None])
        
#         dfs.append(
#             pd.DataFrame({'k': count_cit.keys(), 'n':count_cit.values(), 'pid': pid})
#         )


#     dfs = pd.concat(dfs, axis=0)
#     sns.ecdfplot(x='k', data=dfs)


# ----------------------------------------------------------------------------- #

# def update_paper_from_s2orc():
#     """
#     As we add new papers to a feud, we want to get their citation graph
#     without rerunning those that we already have
#     """
#     # Step 1. Get feud seeds metadata
#     get_seeds_fight_papers()

#     # Step 2. Get citation graph of the feud seeds
#     with jsonlines.open("../data/seeds_fights_papers.jsonl") as f:
#         feud_seeds = f.read()

#     done_citation_graph = set([re.sub(".jsonl", "", str(_).split("/")[-1]) for _ in Path('../feuds').glob("*jsonl")])
#     feuds_to_do = [feud for feud in feud_seeds if get_id_from_url(feud['url']) not in done_citation_graph]

#     for paper in feuds_to_do:
#         get_citation_graph(paper)
#         sleep(10)

#     # Step 3. Get all the papers
#     get_papers()
    
# update_paper_from_s2orc()