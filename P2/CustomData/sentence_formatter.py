import re
import pickle
import os

def change_extension(file_name: str, new_extension: str) -> str:
    # Get the base name of the file (without the extension)
    base_name = os.path.splitext(file_name)[0]
    # Join the base name with the new extension
    new_file_name = base_name + "." + new_extension
    return new_file_name


input_file = 'sentences_32_notsoclean.txt'
formatted_sentences = []

with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        line = re.sub(r"[^a-zA-Z]", "", line)
        line = line.upper()
        assert(len(line) == 32)
        formatted_sentences.append(line)


output_file = change_extension(input_file, 'pkl')
with open(output_file, 'wb') as file:
    pickle.dump(formatted_sentences, file)

print(formatted_sentences)
