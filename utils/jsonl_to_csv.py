import json
import re

def jsonl_to_csv(input_filepath, output_filepath, annotation_type, annotation_type_name):
    with open(input_filepath, 'r', encoding='utf-8') as file:
        jsonl_data = [json.loads(line) for line in file]

    with open(output_filepath, 'w', encoding='utf-8') as file:
        for entry in jsonl_data:
            text = entry['text'].replace('\n', ' ')
            entities = [e for e in entry['entities'] if e['label'] == annotation_type_name]
            entities.sort(key=lambda e: e['start_offset']) #sort by start offset
            
            tokens = re.findall(r"\w+|\w+(?='s)|'s|['\".,!?;]", text, re.UNICODE) #split on spaces, keep punctuation
            
            current_pos = 0
            entity_index = 0
            tags = ''
            for token in tokens:
                start_offset = text.find(token, current_pos)
                end_offset = start_offset + len(token)
                current_pos = end_offset

                tag = 'O'
                if entity_index < len(entities):
                    entity = entities[entity_index]
                    if start_offset == entity['start_offset']:
                        tag = 'B-' + annotation_type
                    elif start_offset > entity['start_offset'] and end_offset <= entity['end_offset']:
                        tag = 'I-' + annotation_type
                    if end_offset == entity['end_offset'] and entity_index < len(entities) - 1:
                        entity_index += 1

                if tags == "":
                    tags = f"{tags}{tag}"
                else:
                    tags = f"{tags} {tag}"

            file.write(f"{text[:-1]}|{tags}\n")

import argparse

parser = argparse.ArgumentParser(
        description='This script is used to convert JSONL data into CSV format.')

parser.add_argument('-o', '--output', type=str, default="all.csv",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-i', '--input', type=str, default="all.jsonl",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Specify the type of annotation to process. Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

args = parser.parse_args()

if not args.output.endswith('.csv'):
    raise ValueError('Output file needs to be defined as a CSV-file')

if not args.input.endswith('.jsonl'):
    raise ValueError('Input file needs to be defined as a JSONL-file')

if args.type not in ['Medical Condition', 'Symptom', 'Medication', 'Vital Statistic', 'Measurement Value', 'Negation Cue', 'Medical Procedure']:
    raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

if args.type == 'Medical Condition':
    annotation_type = 'MEDCOND'
    annotation_type_name = 'Medical Condition'
elif args.type == 'Symptom':
    annotation_type = 'SYMPTOM'
    annotation_type_name = 'Symptom'
elif args.type == 'Medication':
    annotation_type = 'MEDICATION'
    annotation_type_name = 'Medication/Treatment'
elif args.type == 'Vital Statistic':
    annotation_type = 'VITALSTAT'
    annotation_type_name = 'Vital Statistic'
elif args.type == 'Measurement Value':
    annotation_type = 'MEASVAL'
    annotation_type_name = 'Measurement Value'
elif args.type == 'Negation Cue':
    annotation_type = 'NEGATION'
    annotation_type_name = 'Negation Cue'
elif args.type == 'Medical Procedure':
    annotation_type = 'PROCEDURE'
    annotation_type_name = 'Medical Procedure'
else:    
    raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

jsonl_to_csv(args.input, args.output, annotation_type, annotation_type_name)

print(f"Conversion of {args.input} to {args.output} for type \"{annotation_type_name}\" completed.")
