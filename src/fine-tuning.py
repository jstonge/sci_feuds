import json
import openai
import tiktoken
from pathlib import Path
import re
import spacy
import pandas as pd

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'

openai.api_key = open("apikey.txt", "r").read().strip("\n")

def flatten(l):
    return [item for sublist in l for item in sublist]


# ----------------------------- stance detection ----------------------------- #

import pandas as pd
import tiktoken
import seaborn as sns
import openai
import re
import pandas as pd

openai.api_key = open("apikey.txt", "r").read().strip("\n")
enc = tiktoken.encoding_for_model("ada")

def map2int_5(x):
    if x <= -0.6:
        return 1
    elif x <= -0.2:
        return 2
    elif x <= 0.2:
        return 3
    elif x <= 0.6:
        return 4
    else:
        return 5

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

# 1.load and wrangle
df = pd.read_json("../output/all.json")
df.stance = round(df.stance, 2)
df['stance_discrete'] = df.stance.map(map2int_5)

instructions="What is the stance of the following paragraph?\nGive your answer in the range from -1 to 1 (-1=very negative stance; 1=very positive stance; 0=neutral stance). Intuitively,  the more negative/positive statements an abstract contains, the more negative/ positive it is. The severity of the statements amplify the sentiments."

# 1.2 Check if with prompts it works
def check_baseline():
    dff = df.query('(stance > 0.5 and stance < 1.) or stance < -0.5').reset_index(drop=True)
    example_1, comp_1 = dff.abstract[3], dff.stance[3]
    example_2, comp_2 = dff.abstract[8], dff.stance[8]
    rdm_idx = dff.sample(1).index
    text = dff.abstract[rdm_idx].tolist()[0]
    completions = dff.stance[rdm_idx].tolist()[0]

    prompt = ""
    prompt += f"{instructions}##\n\n{example_1} --> {comp_1}\n\n"
    prompt += f"##\n\n{example_2} --> {comp_2}\n\n"
    prompt += f"##\n\n{text} --> \n\n"
    message_history=[{"role": "assistant", "content": prompt}]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=message_history, 
        temperature=0., 
        max_tokens=1)

    reply_content = completion.choices[0].message.content
    print(f"true:{completions},  reply:{int(reply_content)}, diff={abs(completions - int(reply_content))}\n")
    print(text)

check_baseline()

# 1.2 wrangle
prompt = [f"{instructions}\n\nParagraph: {text}\n\n###\n\n" for text in df.abstract]
completion = df['stance_discrete'].map(lambda x: " " + str(round(x, 2)))
df_prompt_completion = pd.DataFrame(zip(prompt, completion), columns=["prompt", "completion"])

# 2. Longer abstracts are noisy and costly to fine-tune, lets get rid of them.
def min_wc(thresh=550):
    return df_prompt_completion[df_prompt_completion.prompt.str.split(" ").map(len) < thresh]

df_prompt_completion = min_wc()

# 3. Many positive examples, lets get rid of some of them so that we have a more balanced dataset
# THRESH_PROMPT_COUNTS = 255
# df_prompt_completion.value_counts("completion")
# df_prompt_completion = df_prompt_completion.sample(frac=1).groupby("completion").head(THRESH_PROMPT_COUNTS)
# df_prompt_completion.value_counts("completion")

# 4. Write data to fine tune
# len(df_prompt_completion.value_counts("completion"))
df_prompt_completion.to_json("../output/stance_detection.jsonl", lines=True, orient='records')

# 4.5. Prepare and fine-tune data using openAI API

# 5. Inspect prepared data
prep_train = pd.read_json("../output/stance_detection_prepared_train.jsonl", lines=True, orient="records")

# 6. Validating fine-tuned model
# !openai wandb sync

model_5 = 'ft-SYIux6qjwOhCzpuOvehWkZx3'
model_9 = 'ft-Twd7JqVUKubIFg5zyK0Mn4Q0'

res = pd.read_csv(f"../output/classif_metric_{model_5}.csv")

res[res['classification/accuracy'].notnull()]['classification/accuracy'].plot() # not great
res[res['classification/weighted_f1_score'].notnull()]['classification/weighted_f1_score'].plot() # not great

# df_long = pd.melt(results[['step', 'training_loss', 'validation_loss']], id_vars=['step'], value_vars=['training_loss', 'validation_loss'])
# sns.lineplot(x='step', y='value', hue='variable', data=df_long[~df_long.value.isna()])

# 7. trying out my model
mydf=pd.read_csv("../output/spacy_group_selection_grobid/par_w_ent_1960_2023.csv")
instructions="What is the stance of the following paragraph?\nGive your answer in the range from -1 to 1 (-1=very negative stance; 1=very positive stance; 0=neutral stance). Intuitively,  the more negative/positive statements an abstract contains, the more negative/ positive it is. The severity of the statements amplify the sentiments."
text = mydf.sentence[2]

model_5 = 'ada:ft-personal-2023-05-01-17-51-22'
model_9 = 'ada:ft-personal-2023-05-01-15-22-13'
res = openai.Completion.create(model="ada:ft-personal-2023-05-01-15-22-13",prompt=text + "\n\n###\n\n", max_tokens=1, temperature=0, logprobs=2)

int(res.choices[0]['text'][1])
res.choices[0]['logprobs']['top_logprobs'][0]


# ------------------------------------ CCA ----------------------------------- #



df_5 = pd.concat([pd.read_json(f"../output/{x}_5.json") for x in ['train', 'test', 'dev']], axis=0)
df_gold = pd.concat([pd.read_json(f"../output/{x}_gold.json") for x in ['train', 'test', 'dev']], axis=0)
df_combined = df_5.merge(df_gold, on='id', how='left', suffixes=['_5', '_gold'])

prompt = df_combined.x_5.map(lambda x: str(x) + "\n\n###\n\n")
completion = df_combined.x_gold.map(lambda x: " " + str(x) + "END")

df_prompt_completion = pd.DataFrame(zip(prompt, completion), columns=['prompt', 'completion']).sample(2000)
# df_prompt_completion = df_combined[df_combined.id.isin(df_todo.id)]
# df_prompt_completion = df_prompt_completion[df_prompt_completion.x_gold.str.count(" ") > 40]
# df_prompt_completion = df_prompt_completion.sample(2000)
df_prompt_completion.to_json("../output/context_detection3.jsonl", lines=True, orient='records')

# df_prompt_completion.to_csv(f"../output/.cache_fine_tuned/third_iteration_cca_{str(date.today())}.csv")

# Validating the model
df_prompt_completion_1 = pd.read_csv(f"../output/.cache_fine_tuned/first_iteration_cca_2023-05-01.csv", names=['index', 'prompt', 'completion'], skiprows=1, index_col='index')
df_prompt_completion_2 = pd.read_csv(f"../output/.cache_fine_tuned/second_iteration_cca_2023-05-01.csv", names=['index', 'prompt', 'completion'], skiprows=1, index_col='index')
df_prompt_completion = pd.concat([df_prompt_completion_1, df_prompt_completion_2], axis=0)

done_idx = df_combined.loc[df_prompt_completion.index, :].id
df_todo = df_combined[~df_combined.id.isin(done_idx)]

model_name  = "curie:ft-personal-2023-05-01-19-28-15"
model_name2 = "curie:ft-personal-2023-05-01-22-25-51"
model_name3 = "curie:ft-personal-2023-05-01-23-05-22"

def print_example(df, mod, max_wc=100, min_wc=0):
    df = df[(df.x_gold.str.count(" ") >= min_wc) & (df.x_gold.str.count(" ") <= max_wc)]
    # rdm_idx = df.sample(1).index[0]
    rdm_idx = 6901
    
    test1 = " However, convolutional models must be significantly deeper to retrieve the same temporal receptive field [23] . Recently, the mechanism of self-attention<cite> [22,</cite> 24] was proposed, which uses the whole sequence at once to model feature interactions that are arbitrarily distant in time. Its use in both encoder-decoder and feedforward contexts has led to faster training and state-of-the-art results in translation (via the Transformer<cite> [22]</cite> ), sentiment analysis [25] , and other tasks. These successes have motivated preliminary work in self-attention for ASR. Time-restricted self-attention was used as a drop-in replacement for individual layers in the state-of-theart lattice-free MMI model [26] , an HMM-NN system."
    target_x = "Its use in both encoder-decoder and feedforward contexts has led to faster training and state-of-the-art results in translation (via the Transformer<cite> [22]</cite> ), sentiment analysis [25] , and other tasks."
    # test1 = df_todo[rdm_idx]
    # target_x = df_todo[rdm_idx]

    res = openai.Completion.create(model=mod, prompt=test1 + "\n\n###\n\n", temperature=0, max_tokens=500)
    reply_content=re.findall("^ ?.+?(?=END|$)", res.choices[0]['text'], re.DOTALL)[0].strip()
    print(f"rdm_idx: {rdm_idx}\n\ngiven: {test1}\n\ntarget: {target_x}\n\nreply: {reply_content}")

print_example(df_todo, model_name3, max_wc=40)

import json
# validating with group selection feud
model_name = "curie:ft-personal-2023-05-01-19-28-15"
mydf = pd.read_csv("../output/groupSel_feud.csv")
mydf = mydf[~mydf.abstract.isna()]
mydf = mydf[mydf.abstract.str.contains("(wynne|W\.-?E\.)", case=False)].reset_index(drop=True)
nlp = spacy.load("en_core_web_sm")
doc = nlp(mydf.abstract[0])
sents = []
for sent in doc.sents:
    sents.append(sent.text)
myex = ' '.join(sents[0:5]) + '\n\n##\n\n'
myex = re.sub("Wynne-Edwards", "<cite>Wynne-Edwards</cite>", myex)

mydf=pd.read_json("../data/a97a19ee8eb086df03961634cca804b551cd4a4c.jsonl", lines=True, orient="records")
mydf['content'][0]['text']
bib_test = json.loads(mydf['content'][0]['annotations']['bibref'])[0]
mydf['content'][0]['text'][int(bib_test['start']):int(bib_test['end'])]
myex = re.sub("[11]", "<cite>[11]</cite>", mydf['content'][0]['text'][2115:3074])+"\n\n###\n\n"

res = openai.Completion.create(model=model_name, prompt=myex, temperature=0, max_tokens=500)
reply_content=re.findall("^ ?.+?(?=END|$)", res.choices[0]['text'], re.DOTALL)[0].strip()
print(f"given: {myex}\n\nreply: {reply_content}")



