import json
import re

def jsonl_to_conll(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]

    with open(output_filepath, 'w', encoding='utf-8') as file:
        for entry in jsonl_data:
            text = entry['text'].replace('\n', ' ')
            entities = entry['entities']
            entities.sort(key=lambda e: e['start_offset']) #sort by start offset
            
            tokens = re.findall(r"\w+|\w+(?='s)|'s|['\".,!?;]", text, re.UNICODE) #split on spaces, keep punctuation
            
            current_pos = 0
            entity_index = 0
            for token in tokens:
                start_offset = text.find(token, current_pos)
                end_offset = start_offset + len(token)
                current_pos = end_offset

                tag = 'O'
                if entity_index < len(entities):
                    entity = entities[entity_index]
                    if start_offset == entity['start_offset']:
                        tag = 'B-' + 'MEDCOND'
                    elif start_offset > entity['start_offset'] and end_offset <= entity['end_offset']:
                        tag = 'I-' + 'MEDCOND'
                    if end_offset == entity['end_offset'] and entity_index < len(entities) - 1:
                        entity_index += 1
                    file.write(f"{token} {tag}\n")

            file.write("\n")

jsonl_filepath = '../datasets/labelled_data/all.jsonl'
conll_filepath = '../datasets/labelled_data/all.conll'

jsonl_to_conll(jsonl_filepath, conll_filepath)
