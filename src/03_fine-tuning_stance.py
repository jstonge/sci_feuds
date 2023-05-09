import re
from pathlib import Path

import openai
import pandas as pd
import tiktoken

from helpers import map2int_3

ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / 'output'
GROBID_DIR = OUTPUT_DIR / 'group_selection_grobid'
SPACY_DIR = OUTPUT_DIR / 'spacy_group_selection_grobid'
STANCE_DIR = OUTPUT_DIR / 'stance_detection'
CCA_DIR = OUTPUT_DIR / 'cca'

openai.api_key = open("myapikey.txt", "r").read().strip("\n")

def parse_reply(x):
    return re.findall("^ ?.+?(?=END|$)", x.choices[0]['text'], re.DOTALL)[0].strip()

enc = tiktoken.encoding_for_model("curie")

# 1.load and wrangle
df = pd.read_json("../output/stance_detection/all.json")

df['stance_discrete'] = df.stance.map(map2int_3)

df = df[((df.stance > 0.75) | (df.stance < -0.25)) | ((df.stance > -.25) & (df.stance < .25) ) ]

# df.stance = round(df.stance, 2)
# instructions="What is the stance of the following paragraph?\nGive your answer in the range from -1 to 1 (-1=very negative stance; 1=very positive stance; 0=neutral stance). Intuitively,  the more negative/positive statements an abstract contains, the more negative/ positive it is. The severity of the statements amplify the sentiments."

# 1.2 Check if with prompts it works

# def check_baseline():
#     dff = df.query('(stance > 0.5 and stance < 1.) or stance < -0.5').reset_index(drop=True)
#     example_1, comp_1 = dff.abstract[3], dff.stance[3]
#     example_2, comp_2 = dff.abstract[8], dff.stance[8]
#     rdm_idx = dff.sample(1).index
#     text = dff.abstract[rdm_idx].tolist()[0]
#     completions = dff.stance[rdm_idx].tolist()[0]

#     prompt = ""
#     prompt += f"{instructions}##\n\n{example_1} --> {comp_1}\n\n"
#     prompt += f"##\n\n{example_2} --> {comp_2}\n\n"
#     prompt += f"##\n\n{text} --> \n\n"
#     message_history=[{"role": "assistant", "content": prompt}]

#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo", 
#         messages=message_history, 
#         temperature=0., 
#         max_tokens=1)

#     reply_content = completion.choices[0].message.content
#     print(f"true:{completions},  reply:{int(reply_content)}, diff={abs(completions - int(reply_content))}\n")
#     print(text)

# check_baseline()

# 1.2 wrangle


prompt = df.abstract.map(lambda x: str(x) + "\n\n###\n\n")
completion = df['stance_discrete'].map(lambda x: " " + x)
df_prompt_completion = pd.DataFrame(zip(prompt, completion), columns=["prompt", "completion"])

#Get rid of long paragraphs.
def min_wc(thresh=550):
    return df_prompt_completion[df_prompt_completion.prompt.str.split(" ").map(len) < thresh]

df_prompt_completion = min_wc()

# Get rid of some positive examples so that we have a more balanced dataset
# THRESH_PROMPT_COUNTS = 230
# df_prompt_completion.value_counts("completion")
# df_prompt_completion = df_prompt_completion.sample(frac=1).groupby("completion").head(THRESH_PROMPT_COUNTS)
# df_prompt_completion.value_counts("completion")

# 4. Write data to fine tune
# df_prompt_completion.value_counts("completion")
df_prompt_completion.to_json(STANCE_DIR / "stance_detection_2023-05-08.jsonl", lines=True, orient='records')

# 4.5. Prepare and fine-tune data using openAI API

# 5. Inspect prepared data
df_train = pd.read_json(STANCE_DIR / "stance_detection_2023-05-08_prepared_train.jsonl", lines=True, orient="records")
df_valid = pd.read_json(STANCE_DIR / "stance_detection_2023-05-08_prepared_valid.jsonl", lines=True, orient="records")

df_train.value_counts("completion")
df_valid.value_counts("completion")

# file_name = STANCE_DIR / "stance_detection_2023-05-08.jsonl"

# upload_response = openai.File.create(
#   file=open(file_name, "rb"),
#   purpose='fine-tune',
#   api_key=openai.api_key
# )

# file_id = upload_response.id

# response = openai.FineTune.create(training_file=file_id, model="curie")




# 6. Validating fine-tuned model
# !openai wandb sync

best_model_so_far = "curie:ft-personal-2023-05-08-19-17-01"
best_model_so_far = "curie:ft-personal-2023-05-08-21-10-03"

# res = pd.read_csv(f"../output/classif_metric_{model_5}.csv")

# res[res['classification/accuracy'].notnull()]['classification/accuracy'].plot() # not great
# res[res['classification/weighted_f1_score'].notnull()]['classification/weighted_f1_score'].plot() # not great

# df_long = pd.melt(results[['step', 'training_loss', 'validation_loss']], id_vars=['step'], value_vars=['training_loss', 'validation_loss'])
# sns.lineplot(x='step', y='value', hue='variable', data=df_long[~df_long.value.isna()])

# 7. trying out model
df_sent = pd.read_json(STANCE_DIR / 'My-Predictions.json')
# stance_pred = df_sent[df_sent.stance <= -0.2].stance[10]
# stance_pred = df_sent[df_sent.stance <= -0.2].stance[10]
text = df_sent[df_sent.stance <= -0.2].abstract[10]

# mydf = pd.read_parquet("groupSel_fine_tuned_cca.parquet")
# mydf=pd.read_csv("../output/spacy_group_selection_grobid/par_w_ent_1960_2023.csv")
# instructions="What is the stance of the following paragraph?\nGive your answer in the range from -1 to 1 (-1=very negative stance; 1=very positive stance; 0=neutral stance). Intuitively,  the more negative/positive statements an abstract contains, the more negative/ positive it is. The severity of the statements amplify the sentiments."
# text=mydf.parsed_doc[170]

# model_5 = 'ada:ft-personal-2023-05-01-17-51-22'
# model_9 = 'ada:ft-personal-2023-05-01-15-22-13'

res = openai.Completion.create(model=best_model_so_far, prompt=text + "\n\n###\n\n", max_tokens=1, temperature=0, logprobs=2)

reply = parse_reply(res)
# res.choices[0]['logprobs']['top_logprobs'][0]
print(f"**Prediction curie:** {reply}\n\n**Prediction scibert:** {stance_pred}\n\n**text:** {text}")
