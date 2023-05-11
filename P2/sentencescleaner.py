import pandas as pd
import pickle
import re

path = "P2/Data/dictionary.csv"

def clean_string(input_string):
    cleaned_string = re.sub('[^a-zA-Z]', '', input_string)
    return cleaned_string

df = pd.read_csv(path, header = None)
df.columns = ['word', 'type', 'definition']

clean = []

for index, row in df.iterrows():
    parsed = clean_string(row['definition'])
    if len(parsed) >= 32:
        for i in range(0, int(len(parsed)/32)):
            sentence = parsed[i*32:(i+1)*32].upper()
            clean.append(sentence)

with open(f"P2/data/our_training.pkl", 'wb') as file:
    pickle.dump(clean, file)
