import re

text = None
input_file = 'raw_text.txt'

print(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()
print(f"{input_file} succesfully read.")

pattern_quantity = r"^[A-Z]\s?(?:[a-z][\s']?){31}[,.](?!\w)"
pattern_quality = r"^[A-Z]\s?(?:[a-z][\s']?){31}[.](?!\w)"
sentence_32 = re.compile(pattern_quantity, flags=re.MULTILINE)

print("\nSearching for patterns in text...")
matches = re.findall(sentence_32, text)
print(f"{len(matches)} matches found.")

print("\nWriting results...")
with open('sentences_32_notsoclean.txt', 'w', encoding='utf-8') as file:
    for match in matches:
        file.write(match)
        file.write('\n')