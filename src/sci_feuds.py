import requests
from jsonlines import jsonlines
from pathlib import Path
import sys
from bson import json_util
import json
import shutil
import re
from textwrap import wrap
from creds import client
import bibtexparser
import pandas as pd


from selenium import webdriver
from selenium.webdriver.common.by import By
import os
from io import BytesIO
import zipfile
import base64

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


from time import sleep

sys.path.append("../../s2orc_helpers/s2orc_helpers")

from s2orc_helpers import parse_res, parse_bibref

ROOT_DIR = Path("../")
PDF_DIR = ROOT_DIR / 'data' / 'raw_pdfs'


# ---------------------------------- helpers --------------------------------- #


def flatten(l):
    return [item for sublist in l for item in sublist]

def read_bibtex(fname):
    with open(fname) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database

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

def get_id_from_url(x):
    return re.sub("https://www.semanticscholar.org/paper/", "", x)

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

def install_addon(self, path, temporary=False) -> str:
    """Installs Firefox addon."""

    if os.path.isdir(path):
        fp = BytesIO()
        path_root = len(path) + 1  # account for trailing slash
        with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zipped:
            for base, dirs, files in os.walk(path):
                for fyle in files:
                    filename = os.path.join(base, fyle)
                    zipped.write(filename, filename[path_root:])
        addon = base64.b64encode(fp.getvalue()).decode("UTF-8")
    else:
        with open(path, "rb") as file:
            addon = base64.b64encode(file.read()).decode("UTF-8")

    payload = {"addon": addon, "temporary": temporary}
    return self.execute("INSTALL_ADDON", payload)["value"]


# ----------------------------------------------------------------------------- #

def update_paper_from_s2orc():
    """
    As we add new papers to a feud, we want to get their citation graph
    without rerunning those that we already have
    """
    # Step 1. Get feud seeds metadata
    get_seeds_fight_papers()

    # Step 2. Get citation graph of the feud seeds
    with jsonlines.open("../data/seeds_fights_papers.jsonl") as f:
        feud_seeds = f.read()

    done_citation_graph = set([re.sub(".jsonl", "", str(_).split("/")[-1]) for _ in Path('../feuds').glob("*jsonl")])
    feuds_to_do = [feud for feud in feud_seeds if get_id_from_url(feud['url']) not in done_citation_graph]

    for paper in feuds_to_do:
        get_citation_graph(paper)
        sleep(10)

    # Step 3. Get all the papers
    get_papers()

update_paper_from_s2orc()


# -------------------- Scrapping group selection feud ------------------------- #


# Since paper coverage for group selection feud is not great, we need to scrape
# the PDFs and parse them. This is time consuming and we do it semi-manually.


with jsonlines.open("../data/seeds_fights_papers.jsonl") as f:
    feud_seeds = f.read()

all_pids = [get_id_from_url(f['url']) for f in feud_seeds]

pid='9bec2ca8875bf7179e6b3371b5328a6af5af55de' # group selection pid
target_paper = [_ for _ in feud_seeds if get_id_from_url(_['url']) == pid]
meta_papers=read_meta_papers(pid)
len(meta_papers) # we know about 1875 papers for group selection feud

urls2get =  ["https://www.semanticscholar.org/paper/"+p['citingPaper']['paperId'] for p in meta_papers if p['citingPaper'].get('paperId')]


driver = webdriver.Firefox()

driver.install_addon("../dl?browser=firefox&version=5.0.107", temporary=True)
driver.get("about:support")
addons = driver.find_element(By.XPATH, '//*[contains(text(),"Add-ons") and not(contains(text(),"with"))]')
driver.execute_script("arguments[0].scrollIntoView();", addons)

driver.get(urls2get[1872])
driver.quit()


# ----------------------------- checking the data ---------------------------- #


bibdatabase = read_bibtex('../data/group_selection.bib')

# Some things to look at
# Pdfs duplicated by entry 

dup_pdfs = [e['file'] if e.get('file') else None for e in bibdatabase.entries 
            if e.get('file') 
            and len(re.findall("/home/jstonge/Zotero/storage/\w+", e['file'])) > 1]



# Entries without paperIds, or wrong value.
# {_['ID']: _['annote'] for _ in bibdatabase.entries if len(_['annote']) != 50}

# Duplicated titles
all_titles = [entry['title'].lower() for entry in bibdatabase.entries]
titles_counts = Counter(all_titles)
{k:v for k,v in titles_counts.items() if v != 1}

# Extracting Citation
# Books have parsing other than citation counts in extra
# no_cit_count = [entry for entry in bibdatabase.entries if entry.get('note') and len(entry.get('note')) == 1]

def create_df_from_bib():
    ids = [e['ID'] for e in bibdatabase.entries]
    titles = [e['title'] for e in bibdatabase.entries]
    years = [f"{e['year']}-01-01" for e in bibdatabase.entries]
    abstracts = ["\n".join(wrap(e['abstract'], width=200)) if e.get('abstract') else None for e in bibdatabase.entries]
    cit_counts = [int(re.findall("^\d+", entry['note'])[0]) if entry.get('note') and len(re.findall("^\d+", entry['note'])) > 0 else 0  for entry in bibdatabase.entries]
    authors = [e.get('author') for e in bibdatabase.entries]
    pd.DataFrame({'citationCounts': cit_counts, 'ID': ids, 'title': titles, 'year': years, 'abstract': abstracts, 'author': authors}).to_csv("groupSel_feud.csv", index=False)

create_df_from_bib()

# EDA - plot to do
# - [x] Plotting feud overall pattern based only on citation graph
# - [ ] coverage by decade: bar plot of how many article articles we have with opacity indicating the number of parsed text we have
# - [ ] field of studies/journals: fos participating to the debate

# Once we have each mention + their scores
# - [ ] Pairwise interactions (plot distribution of valence)


# ------------------------------- Parsing pdfs ------------------------------- #


def mv_and_rename_pdfs_in_proj():
    """
    For some entry, we have multiple pdfs. We simply grab the first one at
    the moment.
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


# --------------------- Examining results at paper level ---------------------------- #


with jsonlines.open("../data/seeds_fights_papers.jsonl") as f:
    feud_seeds = f.read()

all_pids = [get_id_from_url(f['url']) for f in feud_seeds]

pid='9bec2ca8875bf7179e6b3371b5328a6af5af55de'
target_paper = [_ for _ in feud_seeds if get_id_from_url(_['url']) == pid]
meta_papers=read_meta_papers(pid)
papers=read_papers(pid)
all_citations = [parse_bibref(p, meta_papers, target_paper[0]) for p in papers]

print_summary()

def plot_all():
    dfs = []
    
    for pid in all_pids:
        print(pid)
        target_paper = [_ for _ in feud_seeds if get_id_from_url(_['url']) == pid]
        meta_papers=read_meta_papers(pid)
        papers=read_papers(pid)
        
        all_citations = [parse_bibref(p, meta_papers, target_paper[0]) for p in papers]
        
        # [f"Paper {i} cites {len(cit['citations'])}" for i, cit in enumerate(all_citations) if cit is not None and cit.get('citations')]

        count_cit = Counter([len(c['citations']) for c in all_citations if c is not None])
        
        dfs.append(
            pd.DataFrame({'k': count_cit.keys(), 'n':count_cit.values(), 'pid': pid})
        )


    dfs = pd.concat(dfs, axis=0)
    sns.ecdfplot(x='k', data=dfs)
