import re
from pathlib import Path

import numpy as np
import openai
import pandas as pd
# import spacy
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DIR   = Path("..")
FIG_DIR    = ROOT_DIR / "figs"
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR  = OUTPUT_DIR / 'spacy_group_selection_grobid'
STANCE_DIR = OUTPUT_DIR / 'stance_detection'
CCA_DIR    = OUTPUT_DIR / 'cca'

pop_authors = set(pd.read_csv(OUTPUT_DIR / "list_hotshots.csv").cite_spans)



def parse_reply(x):
    return re.findall("^ ?.+?(?=END|$)", x.choices[0]['text'], re.DOTALL)[0].strip()

def call_openai(x, model, max_tok=700):
    return openai.Completion.create(
        model=model, prompt=x + "\n\n###\n\n", temperature=0, max_tokens=max_tok
        )


# ------------------------------- cca analysis ------------------------------- #


step1_best_so_far = "curie:ft-personal-2023-05-01-19-28-15"

# How many sentence is a context?


df = pd.read_parquet(CCA_DIR/"groupSel_feud_with_tag.pqt")
df = df[~df.parsed_doc.duplicated()]
df["wc"] = df.parsed_doc.str.count(" ")

we_df = df[df.cite_spans.str.contains("Wynne")]


we_df = we_df[we_df['wc'] < 500]


we_prompts = we_df.parsed_doc.map(lambda x: call_openai(x, step1_best_so_far))


example = we_prompts.map(parse_reply)

we_df['relevant_contexts'] = example

we_df.to_parquet("groupSel_fine_tuned_cca.parquet", index=False)


def show_examples():
    tmp_df = we_df.reset_index()
    rdm_idx = tmp_df.sample(1)['index']
    print(f"given: {tmp_df.parsed_doc[rdm_idx].tolist()}\n")
    print(f"reply: {tmp_df['relevant_contexts'][rdm_idx].tolist()}")

show_examples()

# How many sentence is a context?

d = pd.read_parquet("groupSel_fine_tuned_cca.parquet")

nlp = spacy.load("en_core_web_trf")
docs = list(nlp.pipe(d.relevant_contexts.tolist()))
sent_counts = [len(list(doc.sents)) for doc in docs]

d['sent_counts'] = sent_counts 

f, ax = plt.subplots(1,1,figsize=(5,3))
sns.histplot(sent_counts, ax=ax)
ax.set_xlabel("# sentences")
ax.set_ylabel("frequency")
plt.title("Number of sentences\nfor relevant citation context")
plt.tight_layout()
plt.savefig("../../figs/multicite.pdf")


# --------------------------- cca+stance detection --------------------------- #



# best_model_so_far_5 = "curie:ft-personal-2023-05-09-23-14-24"
best_model_so_far_3 = "curie:ft-personal-2023-05-08-21-10-03"
best_model_so_far_5 = "curie:ft-personal-2023-05-08-19-17-01"
# best_model_so_far_id_5 = "ft-fWXcntKe6ctUxxG2GANyneXJ"
best_model_so_far_id_3 = "ft-F2HCJCsDcXnUodzu37pongJf"
best_model_so_far_id_5 = "ft-FrUEGnWr4iGrIPpuYKDunfJ0"

we_df = pd.read_json(STANCE_DIR / 'My-Predictions.json')

# we_prompts = we_df.abstract.map(lambda x: call_openai(x, best_model_so_far))


res = openai.Completion.create(model=best_model_so_far_3, prompt=text + "\n\n###\n\n", max_tokens=1, temperature=0, logprobs=2)

reply = parse_reply(res)
# res.choices[0]['logprobs']['top_logprobs'][0]
print(f"**Prediction curie:** {reply}\n\n**Prediction scibert:** {stance_pred}\n\n**text:** {text}")
