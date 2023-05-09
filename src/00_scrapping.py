
import base64
import os
import zipfile
from io import BytesIO

from jsonlines import jsonlines
from selenium import webdriver
from selenium.webdriver.common.by import By

from helpers import get_id_from_url, read_meta_papers


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

