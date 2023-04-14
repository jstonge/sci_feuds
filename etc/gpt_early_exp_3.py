
import json
from jsonlines import jsonlines
import openai
import tiktoken
from copy import deepcopy

openai.api_key = open("apikey.txt", "r").read().strip("\n")
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

instructions = """
Instructions:
 - You are an editors working for a journal. 
 - Your task is to analyze articles that consist of many paragraphs.
 - Your analysis should contain the following:
     - Does each sentence contains mentions or references to other authors (recall that names often start with upper case) ?
 - Provide a sentiment score for the paragraphs using the following scale: 1 to 9 (where 1 is most negative, 5 is neutral, and 9 is most positive). 
 - Take into account opposing sentiments in the mentions or references to other authors, but also try to correctly identify descriptive statements. 
 - Format your response in a json format where for each sentence, you provide the text, overall sentiment score, then if there are mentions or references, with their associated sentiment scores, and finally the equation, model, and if this is the author's view.
"""
with open("../output/group_selection_grobid/wiens_behavioral_1965.json") as f:
    dat = json.load(f)

with open("../output/group_selection_grobid/braestrup_animal_1963.json") as f:
    dat = json.load(f)

texts = [_['text'] for _ in dat['pdf_parse']['body_text']]

print(f"Nb tokens: {len(enc.encode(' '.join(texts)))}")

# Define a function to interact with ChatGPT
#!TODO Maybe precalculate everything, then send it once a time to gpt
def chat_with_gpt(max_tokens=1000):
    prompt = ""
    message_history = []
    global_message_history = [] 

    i = 0
    i_last_end = -1
    while i < len(texts):
    
        toks_count = len(enc.encode(prompt)) + len(enc.encode(texts[i]))
        if i == (i_last_end + 1):   # if we are starting a new conversation, prepend the instructions        
            prompt += f"{instructions}\n\n{texts[i]}"
        elif toks_count < max_tokens:
            prompt += f"\n\n{texts[i]}"
        else: #if we have hit the limit

            message_history.append({"role": "user", "content": prompt}) #format the request properly for output

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message_history
            )          
            reply_content = completion.choices[0].message.content
            message_history.append({"role": "assistant", "content": reply_content})

            # update global history
            global_message_history += message_history
            
            # reinitialize everything
            message_history = []
            prompt = ""
            i_last_end = i
        
        i += 1

    return global_message_history

out = chat_with_gpt()



with jsonlines.open("braestrup_1963_gpt35turbo.jsonl", "w") as f:
    f.write(out)


# ------------------------------ comparing text ------------------------------ #

from difflib import SequenceMatcher

with jsonlines.open("braestrup_1963_gpt35turbo.jsonl") as f:
    dat = f.read()


def reconstruct_text_from_reply(text):
    return ' '.join([_['text'] for _ in text])

def flatten(l):
    return [item for sublist in l for item in sublist]

def parse_raw_conv(conv_raw):
    """
      - standardize raw conversation with gpt
    """
    all_conv = []
    for conv in conv_raw:  
        if isinstance(conv, dict) and conv.get('paragraphs'):
            all_conv.append(conv['paragraphs'])
        elif isinstance(conv, dict) and conv.get('sentences'):
            all_conv.append(conv['sentences'])
        elif isinstance(conv, list) and conv[0].get('text'):
            all_conv.append(conv)
    return all_conv

def assess_qual_reconstruction(user, assistant):
    """
     - user: paragraphs in a list given by user
     - assitant: texts in a list given back by gpt
    """
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    assert len(user) == len(assistant), "both list must be of the same len"
    for i in range(4):
        raw_text = ' '.join(user[i])
        print(similar(raw_text, reconstruct_text_from_reply(assistant[i])))

instructions = dat[0]['content'].split("\n\n")[0]
user_content = [_['content'].split("\n\n")[1:] for _ in dat if _['role'] == 'user']
all_conv_raw = [json.loads(conv['content']) for conv in dat if conv['role'] == 'assistant']
all_conv = parse_raw_conv(all_conv_raw)

assess_qual_reconstruction(user_content, all_conv)

all_conv_flatten = flatten(all_conv)

lookup_authors = {'W.-E.': 'Wynne-Edwards',
                  'LACK': 'D. Lack',
                  'Lack': 'D. Lack',
                  'V. C. Wynne-Edwards': 'Wynne-Edwards',
                  'V. C. WYNNE-EDWARDS': 'Wynne-Edwards',
                  'P. L. ERRING-TON': 'P. L. Errington',
                  '0. KALELA': 'Kalela',
                  'WRIGHT': 'Wright',
                  'CARR-SAUNDERS': 'Carr-Saunders',
                  'BREDER and COATES, 1932': 'Breder and Coates',
                  'KARL VON FRISCH': 'Karl Von Frisch',
                  'PFEIFFER, 1962': 'Pfeiffer',
                  'ELLINOR BRO LARSEN': 'Ellinor Bro Larsen',
                  'WESENBERG-LUND 1943': 'Wesenberg-Lund',
                  'BRAESTRUP 1963': 'Braestrup',
                  'Gause': 'Gause'
                  }

author_mentions_raw = [conv.get('mentions') for conv in all_conv_flatten]

def rename_key(x, new_name, old_name):
    x[new_name] = x[old_name]
    del x[old_name]

def clean_author_mentions():
    new_authors_mentions = []
    for authors in author_mentions_raw:
        if authors:
            if len(authors) == 1:
                authors[0]['source'] = 'Braestrup'
                if authors[0].get('name'):
                    rename_key(authors[0], 'target', 'name')
                    new_authors_mentions.append(authors[0])
                elif authors[0].get('text'):
                    rename_key(authors[0], 'target', 'text')
                    new_authors_mentions.append(authors[0])
            if len(authors) > 1:
                for author in authors:
                    author['source'] = 'Braestrup'
                    if author.get('name'):
                        rename_key(author, 'target', 'name')
                        new_authors_mentions.append(author)
                    elif author.get('text'):
                        rename_key(author, 'target', 'text')
                        new_authors_mentions.append(author)

    for auth in new_authors_mentions:
        if auth['target'] and lookup_authors.get(auth['target']):
            auth['target'] = lookup_authors[auth['target']]


clean_author_mentions()


