import spacy
import pandas as pd
from spacy.tokens import DocBin
from tqdm import tqdm
import random
import re
import os
PATH = os.path.dirname(os.path.dirname(__file__))
random.seed(42)

def split_df(df, val_split=0.3):
    ids = list(dict.fromkeys(list(df.id.values)))
    random.shuffle(ids)
    train_ids = round(len(ids)*(1-val_split))
    df_train = df.loc[df.id.isin(ids[:train_ids])]
    df_val = df.loc[df.id.isin(ids[train_ids:])]
    return df_train, df_val

def data_to_spacy_format(df, n_ents_above=6):
    ids = list(dict.fromkeys(list(df.id.values)))
    formating = []
    for i in tqdm(ids):
        df_per_id = df.loc[df.id == i]
        txt_file = open(f"{PATH}/data/train/{i}.txt", "r")
        txt_file = txt_file.read()
        entities = {'entities': []}
        if len(df_per_id) > n_ents_above:
            for index, row in df_per_id.iterrows():
                ent_list = [int(row.discourse_start), int(
                    row.discourse_end), row.discourse_type.upper()]
                entities['entities'].append(ent_list)
            elements = []
            elements.append(txt_file)
            elements.append(entities)
            # select only texts that end with ".", to facilitate parsing.
            if elements[0][-1] == '.':
                formating.append(elements)

    # removes leading and trailing white spaces from entity spans
    invalid_span_tokens = re.compile(r'\s')
    data = []
    for text, annotations in formating:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        data.append([text, {'entities': valid_entities}])

    return data

def make_spacy_model(data):
    """Create spacy model"""
    nlp = spacy.blank("en") # load a new spacy model
    db = DocBin() # create a DocBin object
    for text, annot in tqdm(data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for element in annot["entities"]:
            for start, end, label in [element]: # add character indexes
                span = doc.char_span(start,
                    end,
                    label=label,
                    alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
        doc.ents = ents # label the text with the ents
        db.add(doc)
    return db

def load_results_manual(X_text, model_num = '01', visualize = True):
    """This function ouputs different labels, so the output os a dictionary"""
    nlp = spacy.load(PATH + f"/models/output_model_{model_num}/model-best")
    doc = nlp(X_text)
    if visualize == True:
        spacy.displacy.render(doc, style='ent', jupyter=True)
    ents = doc.ents
    output = []
    for entity in ents:
        row = {'entity': entity.text,
                'label': entity.label_,
                'start': entity.start,
                'end': entity.end}
        output.append(row)
    return output
