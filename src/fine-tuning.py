import json
# import openai
# import tiktoken
import tqdm
from pathlib import Path
import re
from datetime import date
import spacy
from tqdm import tqdm
import pandas as pd

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



import pandas as pd
import json

df=pd.read_csv("../output/spacy_group_selection_grobid/coref_par_w_ent_1960_1970.csv")
par_id = df['uniq_id'].map(lambda x: x.split("', ")[0].split("_")[-2])
article = df['article']
df["article_par_id"] = article+'_'+par_id

with open("../output/project-25-at-2023-04-26-16-26-86cb0a7a.json") as f:
    dat = json.load(f)

def converter(df_res, source_id, target_id):   
    source = df_res[df_res['id'] == source_id].value.iloc[0]
    type = source['labels'][0]
    source = source['text']
    target = df_res[df_res['id'] == target_id].value.iloc[0]
    sentiment = int(target['labels'][0])
    target = target['text']
    return [source, target, sentiment, type]

def convert_all():
    dfs = []
    for d in dat:
        print(d['data']['article'])
        res = d['annotations'][0]['result']
        df_res=pd.DataFrame(res)
        source_ids, target_ids = zip(*[(_['from_id'], _['to_id']) for _ in res 
                                    if _.get('from_id') and _.get('to_id')])
        list_relations = [converter(df_res, source_id, target_id) for source_id, target_id in zip(source_ids, target_ids)]
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





def flatten(l):
    return [item for sublist in l for item in sublist]



import spacy
from fastcoref import spacy_component

df=pd.read_csv("par_w_ent_1960_1970.csv")
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
out=list(nlp.pipe(df.sentence.tolist(), component_cfg={"fastcoref": {'resolve_text': True}}))
df['coref_sent'] = flatten(out)
df['coref_sent'] = df['coref_sent'].map(lambda x: x._.resolved_text)





enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

target_articles = ['braestrup_animal_1963', 'alexander_group_1978']

with open(OUTPUT_DIR / f"group_selection_grobid/{target_articles[0]}.json") as f:
    dat = json.load(f)

texts = [_['text'] for _ in dat['pdf_parse']['body_text']]

instructions = """Extract the sentences in the paragraph from a scientific article below. First extract all sentiment scores (where 1=extremely negative; 5=neutral; 9=extremely positive.), then extract all reference names (other individuals that author cite) with their sentiment score (again from 1 to 9), then extract theoretical  constructs (theories that you could find on wikipedia such as natural selection, population genetics, or group selection) with discrete sentiment score (supportive, descriptive, against).

Desired format:

  { "Sentence 1": [
      "Text":  <text_sentence>,
      "Sentiment": <sentiment_score>,
      "References":  [
       { "Reference name": <reference_name>,  "Reference sentiment": <reference_sentiment> },
       ...
    ],
    "Theory": [
      {"Theory name": <theory>,  "Theory sentiment": <sentiment_theory>},
      ...
   ]
  },
  { "Sentence 2" : [
      "Text":  <text_sentence>,
     ...
  ]
},
  { "Sentence 10" : [
    "Text": <text_sentence>,
  ...
  ] 
}
]
"""

texts[0]




out = [instructions+"\n##\n"+text for text in texts]

completion0 = [
    {
        "Text": "In 1955 V. C. WYNNE-EDWARDS published a paper on low reproductive rates in certain birds.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "V. C. WYNNE-EDWARDS", "Reference sentiment": 5}
        ],
        "Theory": [],   
    },
    {
        "Text": "This paper contained important arguments against views expounded with, what appeared to be strict and irrefutable logic, by LACK (e.g. 1954) , viz., that in birds and mammals the number of eggs laid, or young born, corresponds to the largest number of offspring possible to raise.",
        "Sentiment": 4,
        "References": [
            {"Reference name": "LACK", "Reference sentiment": 3}
        ],
        "Theory": [],   
    },
    {
        "Text": "Lack maintains that if, e.g., some individuals in a bird population (on account of hereditary difference), lay more eggs than others, and thus, on an average, leave a greater number of descendants, the result will be an increase of fertility to the point where the parents are unable to procure enough food for the offspring.",
        "Sentiment": 4,
        "References": [
            {"Reference name": "Lack", "Reference sentiment": 5}
        ],
        "Theory": [],   
    }
    
]

completion1 = [
    {
        "Text": "It may be argued - and this is the line taken by W.-E. - both in the paper mentioned and in the book under reviewthat Lack's view rests on a mistaken notion of natural selection.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "W.-E.", "Reference sentiment": 6},
            {"Reference name": "Lack", "Reference sentiment": 3}
        ],
        "Theory": [
            { "Theory name": "natural selection", "Theory sentiment": "descriptive" }
        ],   
    },
    {
        "Text": "The process of selection works, not only between individuals, but also between groups and between species, thus promoting characters which are to the benefit of the group, even in the face of contrary individual selection which may be kept in check by special devices.",
        "Sentiment": 6,
        "References": [],
        "Theory name": [
            { "Theory name": "process of selection ", "Theory sentiment": "favor" },
            { "Theory name": "individual selection", "Theory sentiment": "favor" }
        ],   
    },
    {
        "Text": "The existence of such evolutionary mechanisms is not only called for by a great many otherwise unaccountable facts, but is also in accordance with genetic evolutionary theory (see e.g. WRIGHT, 1945)",
        "Sentiment": 5,
        "References": [
            {"Reference name": "WRIGHT", "Reference sentiment": 5}
        ],
        "Theory name": [
            {"Theory name": "genetic evolutionary theory", "Theory sentiment": "in favor"}
        ],   
    }
    
]

completion2 = [
    {
        "Text": "Since the bulk of what follows will be in a critical vein, it would be as well to begin by stating that on this main point I am in perfect accordance with Wynne-Edwards.",
        "Sentiment": 6,
        "References": [
            {"Reference name": "Wynne-Edwards", "Reference sentiment": 7},
        ],
        "Theory": [],   
    },
    {
        "Text": "In fact, about 10 years ago, I wrote some books for Scandinavian readers (1952, 1953, and 1954) in which the importance of intergroup selection in connection with many points concerning social adaptation were stressed.",
        "Sentiment": 6,
        "References": [],
        "Theory": [
            { "Theory name": "intergroup selection", "Theory sentiment": "favor" },
            { "Theory name": "social adaptation", "Theory sentiment": "favor" }
        ],   
    },
    {
        "Text": "KALELA, who was concerned with the same problems, wrote, simultanously, notes which were still less accessible to most biologists owing to the fact they were written in Finnish.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "KALELA", "Reference sentiment": 5}
        ],
        "Theory": [],   
    },
    {
        "Text": "However, in 1954 he finally published a paper in German which, I think, is up to now one of the best introductions to the topic.",
        "Sentiment": 6,
        "References": [],
        "Theory name": [],   
    },
    {
        "Text": "It may be recommended in cases where critical readers of Wynne-Edwards\' book, with little previous knowledge of vertebrate 'socio-ecology', have been so scared by wild or even absurd theories that they are inclined to throw the baby out with the bath water.",
        "Sentiment": 4,
        "References": [
            {"Reference name": "Wynne-Edwards", "Reference sentiment": 5}
        ],
        "Theory": [
            { "Theory name": "socio-ecology'", "Theory sentiment": "descriptive" }
        ],   
    },
]

completion3 = [
    {
        "Text": "Among biologists who have laid the foundation for the notion of auto-regulating mechanisms, it is easy for Scandinavians to remember our good friend P. L. ERRING-TON, who has again and again, especially in connection with his life-long work on the muskratstressed the decisive influence of the mutual tolerance or aggression of the animals themselves, in regulating numbers.",
        "Sentiment": 6,
        "References": [
            {"Reference name": "P. L. ERRINGTON", "Reference sentiment": 7}
        ],
        "Theory": [
            { "Theory name": "auto-regulating mechanisms'", "Theory sentiment": "descriptive" }
        ],   
    },
    {
        "Text": "His work is only casually mentioned by W.-E. On the whole, considering the enormous size of the book and the many rather peripheral topics treated, it is surprising how incompletely the really pertinent points are reviewed.",
        "Sentiment": 3,
        "References": [
            {"Reference name": "W.-E.", "Reference sentiment": 4},
        ],
        "Theory": [],   
    },
    {
        "Text": "I think it may be said of the book, with a certain amount of truth, that most of what is sound in it is not new, and most of which is new is not sound.",
        "Sentiment": 3,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "This in itself might not have been a very serious objection.",
        "Sentiment": 4,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "If it had contained a good, well-digested, and balanced summary of facts and views on animals density regulation in relation to social behaviour, together with a clear statement of the problems (and on this foundation a statement of the more extreme theories of the author), it might have been a highly useful book.",
        "Sentiment": 3,
        "References": [],
        "Theory name": [
            { "Theory name": "animals density regulation", "Theory sentiment": "descriptive" },
            { "Theory name": "social behaviour", "Theory sentiment": "descriptive" }
        ],   
    }
]

completion4 = [
    {
        "Text": "Instead, it is propagandistic with all the evil consequences of misplaced enthusiasm.",
        "Sentiment": 3,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "It may of course be read in a leisurely way for the sake of the many interesting details given, but the reader who is eager to discover what exactly the author's theories are, and what evidence he uses to base them on, will find himself frustrated and irritated long before he is through with it.",
        "Sentiment": 3,
        "References": [],
        "Theory": [],   
    }
]

completion5 = [
    {
        "Text": "Various problems and aspects are mixed together in the most confusing way.",
        "Sentiment": 3,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "One point to which I shall revert later is the ambiguous use of the word dispersion in some chapters.",
        "Sentiment": 3,
        "References": [],
        "Theory": [
            { "Theory name": "dispersion", "Theory sentiment": "descriptive" },
        ],   
    },
    {
        "Text": "Usage in other chapters would appear to be in order.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "It is amusing to reflect that if this had been a religious text in the strict sense, philologists might, in some distant future, have referred these chapters to different authors",
        "Sentiment": 4,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "The term epideictic displays, 'signifying literally\'meant for display\' but connoting in its original Greek form the presenting of a sample', is used consistently instead of such commonly used uncommittal terms as communal display, thus perhaps suggesting to the reader, not to mention the author, that the highly controversial theories involved have actually been proved.",
        "Sentiment": 3,
        "References": [],
        "Theory": [
            { "Theory name": "epideictic displays", "Theory sentiment": "descriptive" },
        ],   
    }
]

completion6 = [
    {
        "Text": "The author starts off with a rather convincing statement of the fundamental problem.",
        "Sentiment": 6,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "He points out that there must be an optimum density of each species. Growth of a predator-population above a certain level can lead to over-exploitation of its resources, exemplified by the ravages caused by human 'overfishing'",
        "Sentiment": 5,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "It is characteristic that the exploitation of a stock of whales or fish may still be very profitable even at a time when the stock begins to suffer depletion.",
        "Sentiment": 6,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "Before the pressure of the hunt decreases owing to diminished returns, the stock may already be lastingly damaged (northern right whales!).",
        "Sentiment": 4,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "In any case the productivity will decline. The remedy is to determine the approximate optimum yield and limit the effort to match it ('fish less and catch more').",
        "Sentiment": 4,
        "References": [],
        "Theory": [],   
    }
]

completion7 = [
    {
        "Text": "The author goes on to say that in general the same applies as forcibly to herbivores as to carnivores.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "Now, while this may of course, be broadly true, civilized man\'s exploitation by means of highly perfected mechanical devices has, on the other hand, certain peculiar traits.",
        "Sentiment": 6,
        "References": [],
        "Theory": [],   
    },
    {
        "Text": "Further, I think too little notice is taken of the fact that at least some species are prevented from over-exploitation of resources by limiting factors other than \'intrinsic\' ones, and, finally, it must be remembered that the problem is greatly complicated by the sharing of the same resources by several species.",
        "Sentiment": 4,
        "References": [],
        "Theory": [],   
    },
    {
        "Text":  "The author has provided an answer to the last point, but I do not think it is a correct one (see below).",
        "Sentiment": 3,
        "References": [],
        "Theory": [],   
    }
]

completion8 = [
    {
        "Text": "In any case, the probability, or even certainty, that auto-regulation of numbers is in the interest of species, is of value merely as an inducement to look for such devices.",
        "Sentiment": 5,
        "References": [],
        "Theory": []
    },
    {
        "Text": "The author has brought together much good evidence, but his enthusiasm has, I think, very often carried him too far and it has given him a peculiarly one-sided outlook.",
        "Sentiment": 2,
        "References": [],
        "Theory": []
    }

]

completion9 = [
    {
        "Text": "As far as primitive man is concerned the notion of optimum numbers, auto-regulation of populations, and even the importance of intergroup selection, was already formulated by CARR-SAUNDERS in 1922 in a book which is fully cited by Wynne-Edwards.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "CARR-SAUNDERS", "Reference sentiment": 6},
            {"Reference name": "Wynne-Edwards", "Reference sentiment": 5}
        ],
        "Theory": [
            {"Theory name": "intergroup selection", "Theory sentiment": "descriptive"},
        ]
    },
    {
        "Text": "Carr-Saunders pointed out that territoriality is of great importance even among the most primitive races of mankind.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "Carr-Saunders", "Reference sentiment": 6}
        ],
        "Theory": []
    }

]

completion10 = [
    {
        "Text": "Nowadays it is well known that 'private ownership of land' is a widespread phenomenon, especially among birds and mammals.",
        "Sentiment": 6,
        "References": [],
        "Theory": [
            {"Theory name": "private ownership of land", "Theory sentiment": "favor"},
        ]
    },
    {
        "Text": "The role of territoriality in imposing a ceiling on population density is now generally admitted, especially in cases where food for the individual (or pair) in possession is wholly or to a large extent found in the territory.",
        "Sentiment": 5,
        "References": [],
        "Theory": [
            {"Theory name": "territoriality", "Theory sentiment": "favor"},
        ]
    },
    {
        "Text": "W.-E. maintains that where there is a 'colonial system of property tenure' the size of the colony may be determined by 'convention', ensuring that only a limited number can breed.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "W.-E.", "Reference sentiment": "4"},
        ],
        "Theory": []
    },
    {
        "Text": "This theory, for which there is rather good evidence, especially in rooks, seems to me one of the most valuable novel suggestions in the book.",
        "Sentiment": 6,
        "References": [],
        "Theory": []
    }
]

completion11 = [
    {
        "Text": "Institutions such as these, and also 'peck order' (the fact that individuals in a flock are generally of unequal status, thus ensuring that some may undergo times of hardship and maintain excellent health while others perish), conform to W.-E.'s contention that actual fighting over food is avoided by 'conventional competition'",
        "Sentiment": 6,
        "References": [
             {"Reference name": "W.-E.", "Reference sentiment": 3},
        ],
        "Theory": [
            {"Theory name": "peck order", "Theory sentiment": "descriptive"},
            {"Theory name": "conventional competition", "Theory sentiment": "against"},
        ],
        "Author view": True   
    },
    {
        "Text": "It may be said that while food or a mate or other requisites inside the habitat constitute the ultimate goal, the proximate goal is territory or social status.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
        "Author view": True   
    },
    {
        "Text": "Furthermore, fighting is usually highly ritualized.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
        "Author view": True   
    },
    {
        "Text": "By this means actual bloodshed, which might be detrimental to the group and to the species in general, is minimized.",
        "Sentiment": 6,
        "References": [],
        "Theory": [],
        "Author view": True   
    },
    {
        "Text": "Ritualization of this kind belongs to social traits of which the origin is incomprehensible without intergroup selection.",
        "Sentiment": 6,
        "References": [],
        "Theory": [
            {"Theory name": "intergroup selection", "Theory sentiment": "favor"},
        ],
        "Author view": True   
    }
]

completion13 = [
    {
        "Text": "The author seems to have arrived at this highly improbable conclusion by a deplorable propensity to include very different phenomena under the same heading.",
        "Sentiment": 3,
        "References": [],
        "Theory": []
    },
    {
        "Text": "At least, he often fails to take account of very deep seated dissimilarities.",
        "Sentiment": 3,
        "References": [],
        "Theory": []
    },
    {
        "Text": "For instance, the follow-my-leader reaction of the processionary caterpillars is mentioned as an example of (inborn) conventional behaviour (p. 449), and on p. 127 we find the following sentences",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
    }
]

completion15 = [
    {
        "Text": "Some of the best evidence comes from experiments with guppies (Lebistes).",
        "Sentiment": 6,
        "References": [],
        "Theory": []
    },
    {
        "Text": "If a colony is started in a tank containing a certain amount of water and with ample food, the final population (biomass) will always be of the same size (about 32 grams fish in 17 litres of water).",
        "Sentiment": 5,
        "References": [],
        "Theory": []
    },
    {
        "Text": "If too many adults were added, some of them died for unknown reasons 'without symptom of disease, injury, or other tangible effect' (BREDER and COATES, 1932, p. 149) , and if, e. g., a single female was added in, the young were devoured after the limit was reached. ",
        "Sentiment": 5,
        "References": [
            {"Reference name": "BREDER and COATES", "Reference sentiment": 5},
        ],
        "Theory": [],
    },
    {
        "Text": "W.-E. writes (p. 543) that the adults too consumed each other in the experiment made by Breder and Coates, but this seems to be a misunderstanding.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "W.-E.", "Reference sentiment": 3},
            {"Reference name": "Breder and Coates", "Reference sentiment": 5},
        ],
        "Theory": [],
    },
    {
        "Text": "There is here a very interesting complication, which seems to have escaped W.-E.\'s attention (this may be just as well, considering what he might have made of it!)",
        "Sentiment": 5,
        "References": [
            {"Reference name": "W.-E.", "Reference sentiment": 2}
        ],
        "Theory": [],
    },
    {
        "Text": "In all Ostariophysi (thus including guppies, and, in fact, most freshwater fishes) the skin contains a substance which, when released into the water (e.g. when one of the school is eaten by a predator), brings about a characteristic reaction of alarm in other members of the same or related species.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
    },
    {
        "Text": "This reaction protects the older fish from predatory attack.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
    },
    {
        "Text": "Since its discovery in 1938 by KARL VON FRISCH this \'Schreckreaktion\' has received a considerable amount of study (recently reviewed by PFEIFFER, 1962 ).",
        "Sentiment": 6,
        "References": [
            {"Reference name": "KARL VON FRISCH", "Reference sentiment": 6},
            {"Reference name": "PFEIFFER", "Reference sentiment": 6},

        ],
        "Theory": [
            {"Theory name": "Schreckreaktion", "Theory sentiment": "in favor"}
        ],
    },
    {
        "Text": "Of course this will also prevent cannibalism, or, at least, will make it improbable in adults.",
        "Sentiment": 5,
        "References": [],
        "Theory": [],
    }
]

completion21 = [
    {
        "Text": "In many cases gatherings (usually with displays) of animals certainly have a special social significance, but I think that the hitherto favoured explanation, viz. that they help to effect the partitioning of the species into more or less discrete populations, is the correct one (to have group-selection we must have groups).",
        "Sentiment": 6,
        "References": [

        ],
        "Theory": [
            {"Theory name": "group-selection", "Theory sentiment": "descriptive"},
        ]
    },
    {
        "Text": "This problem will be treated in a special publication (BRAESTRUP 1963), to which a criticism of W.-E.'s peculiar views concerning the role of the males in regulating fecundity will also be deferred.",
        "Sentiment": 5,
        "References": [
            {"Reference name": "BRAESTRUP", "Reference sentiment": 5},
            {"Reference name": "W.-E.", "Reference sentiment": 3},
        ],
        "Theory": []
    }
]


[
    {"prompt": out[0], "completion": completion0},
    {"prompt": out[1], "completion": completion1},
    {"prompt": out[2], "completion": completion2},
    {"prompt": out[3], "completion": completion3},
    {"prompt": out[4], "completion": completion4},
    {"prompt": out[5], "completion": completion5},
    {"prompt": out[6], "completion": completion6},
    {"prompt": out[7], "completion": completion7},
    {"prompt": out[8], "completion": completion8},
    {"prompt": out[9], "completion": completion9},
    {"prompt": out[10], "completion": completion10},
    {"prompt": out[11], "completion": completion11},
    {"prompt": out[13], "completion": completion13},
    {"prompt": out[15], "completion": completion15},
    {"prompt": out[21], "completion": completion21},
]
