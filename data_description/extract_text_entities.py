import json

def extract_text_entities(input_file_path, text_file_path, entities_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        jsonl_entry = json.loads(line)
        text_content = jsonl_entry['text']
        entities_content = jsonl_entry['entities']

        extracted_entities = [text_content[entity['start_offset']:entity['end_offset']] for entity in entities_content]

        with open(text_file_path, 'a') as text_file:
            text_file.write(text_content)

        with open(entities_file_path, 'a') as entities_file:
            for entity in extracted_entities:
                entities_file.write(entity + '\n')

# Define file paths
input_jsonl_file_path = '../datasets/labelled_data/all.jsonl'
output_text_file_path = './text.txt'
output_entities_file_path = './entities.txt'

extract_text_entities(input_jsonl_file_path, output_text_file_path, output_entities_file_path)
