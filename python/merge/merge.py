import json
import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from alias_prompt import FEW_SHOT_EXTRACT_ALIAS, SYSTEM_PROMPT_EXTRACT_ALIAS
import openai
import tqdm
import pandas as pd
def merge_cell_types(folder_path: str, output_file: str) -> Dict[str, Any]:
   """ Merge JSON files with the same cell_type in the extraction folder."""
   merged_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
      "cell_type": "",
      "extractions": [],
      "extraction_num": 0
   })
   
   for filename in os.listdir(folder_path):
      if not filename.endswith('.json'):
         continue
         
      file_path = os.path.join(folder_path, filename)
      with open(file_path, 'r', encoding='utf-8') as f:
         data = json.load(f)
         
      for entry in data:
         cell_type = entry["cell_type"]
         merged = merged_data[cell_type]
         
         if not merged["cell_type"]:
               merged["cell_type"] = cell_type
         
         extraction = {
            "explanation": entry["explanation"],
            "factoids": entry["factoids"],
            "metadata": entry["metadata"]
         }
         
         merged["extractions"].append(extraction)
         merged["extraction_num"] = len(merged["extractions"])

   result = {
      cell_type: data 
      for cell_type, data in sorted(merged_data.items())
   }
   
   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(list(result.values()), f, indent=4, ensure_ascii=False)
   
   return result

def merge_with_flywire(extraction_file: str, flywire_file: str, output_file: str):
   """Merge extraction JSON with flywire CSV data using left join on cell_type."""
   with open(extraction_file, 'r', encoding='utf-8') as f:
      extraction_data = json.load(f)
   
   flywire_lookup = {}
   with open(flywire_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         cell_type = row['cell_type']
         flywire_lookup[cell_type] = {
            'root_id': row.get('root_id', ''),
            'flow': row.get('flow', ''),
            'super_class': row.get('super_class', ''),
            'class': row.get('class', ''),
            'sub_class': row.get('sub_class', ''),
            'hemibrain_type': row.get('hemibrain_type', ''),
            'hemilineage': row.get('hemilineage', ''),
            'side': row.get('side', ''),
            'nerve_feature': row.get('nerve_feature', ''),
            'found_in_flywire': True
         }
   
   for item in extraction_data:
      cell_type = item['cell_type']
      found_in_flywire = item.get('found_in_flywire', False)
      if cell_type in flywire_lookup and not found_in_flywire:
         item.update(flywire_lookup[cell_type])
         found_in_flywire = True
      else:
         item.update({
            'found_in_flywire': False
         })
   
   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(extraction_data, f, indent=4, ensure_ascii=False)

def statistics_report(file_path: str):
   """Generate a statistics report from the merged JSON file."""
   with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
   
   # Count the number of cells found in flywire
   found_in_flywire_count = sum(1 for item in data if item['found_in_flywire'])
   print(f"Number of cells found in flywire: {found_in_flywire_count}")
   
   # Count the number of cells not found in flywire
   not_found_in_flywire_count = sum(1 for item in data if not item['found_in_flywire']) 
   print(f"Number of cells not found in flywire: {not_found_in_flywire_count}")

def load_alias_prompt():
   """Load the alias prompt."""
   prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT_EXTRACT_ALIAS + "\n\n" + FEW_SHOT_EXTRACT_ALIAS),
      ("user", "{input}")
   ])
   return prompt

def find_alias_pair(file_path: str, alias_file: str):
   """Find alias pairs in the merged JSON file."""
   model_type = "gpt-4o"
   model_kwargs = {"response_format": {"type": "json_object"}} 
   default_config = {"temperature": 0, "frequency_penalty": 0.1}
   rst = []
   from langchain_openai import ChatOpenAI

   # Create model and prompt once outside the loop
   model = ChatOpenAI(
         model=model_type,
         model_kwargs=model_kwargs,
         **default_config,
   )
   prompt = load_alias_prompt()
   chain = prompt | model  # Create the chain once

   with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
   for i, item in enumerate(tqdm.tqdm(data[3460:])):
      cell_type = item['cell_type']
      found_in_flywire = item['found_in_flywire']
      # if not found_in_flywire:
      for each in item['extractions']:
         input_data = {
            "cell_type": cell_type,
            "explanation": each['explanation'],
            "factoids": each['factoids']
         }
         try:
            response = chain.invoke({"input": json.dumps(input_data) + "\n\nPlease respond in JSON format."}).content
            response_json = json.loads(response)
            print(f"i:{i}, response_json:{response_json}")

            if response_json["alias_pairs"] != []:
               rst.append(response_json)
               # Write each response immediately in JSONL format
               with open(alias_file, "a", encoding="utf-8") as f:
                  f.write(json.dumps(response_json, ensure_ascii=False) + "\n")
         except openai.BadRequestError as e:
            print(f"Context length exceeded for cell type: {cell_type}")
            print("Input data:", json.dumps(input_data, indent=2))
            continue

def add_alias_to_merged_data(alias_file: str, merged_file: str, output_file: str):
   """Add alias lists to the merged data file."""
   # Read alias pairs from jsonl file
   alias_dict = defaultdict(list)
   with open(alias_file, 'r', encoding='utf-8') as f:
      for line in f:
         data = json.loads(line)
         cell_type = data['alias_pairs'][0]['main_name']
         alias = data['alias_pairs'][0]['alias']
         if cell_type not in alias_dict:
            alias_dict[cell_type] = []
         alias_dict[cell_type].append(alias)

   with open(merged_file, 'r', encoding='utf-8') as f:
      merged_data = json.load(f)
   
   # Add alias lists to merged data
   for item in merged_data:
      cell_type = item['cell_type']
      item['alias'] = alias_dict.get(cell_type, [])
   
   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(merged_data, f, indent=4, ensure_ascii=False)

def merge_with_flywire_using_alias(merged_alias_file: str, flywire_file: str, output_file: str):
   """Merge data with flywire CSV using both cell_type and aliases for matching."""
   # Read the merged data with aliases
   with open(merged_alias_file, 'r', encoding='utf-8') as f:
      merged_data = json.load(f)
   
   flywire_lookup = {}
   with open(flywire_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         cell_type = row['cell_type']
         flywire_lookup[cell_type] = {
               'root_id': row.get('root_id', ''),
               'flow': row.get('flow', ''),
               'super_class': row.get('super_class', ''),
               'class': row.get('class', ''),
               'sub_class': row.get('sub_class', ''),
               'hemibrain_type': row.get('hemibrain_type', ''),
               'hemilineage': row.get('hemilineage', ''),
               'side': row.get('side', ''),
               'nerve_feature': row.get('nerve_feature', ''),
               'found_in_flywire': True,
               'matched_by': 'direct'  
         }
   
   for item in merged_data:
      cell_type = item['cell_type']
      aliases = item.get('alias', [])
      found_in_flywire = item.get('found_in_flywire', False)
      
      if cell_type in flywire_lookup and not found_in_flywire:
         item.update(flywire_lookup[cell_type])
         continue
         
      if not found_in_flywire:
         for alias in aliases:
            if alias in flywire_lookup:
               flywire_data = flywire_lookup[alias].copy()
               flywire_data['matched_by'] = 'alias'  # Indicate match was through alias
               item.update(flywire_data)
               break
         else:
            item.update({
               'found_in_flywire': False,
               'matched_by': 'none'
            })

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(merged_data, f, indent=4, ensure_ascii=False)


def merge_with_community_label(extraction_file, consolidated_file, community_file, output_file):
   """Merge extraction JSON with consolidated JSON using left join on cell_type."""
   additional_data = {}
   with open(consolidated_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         if row['additional_type(s)'] != '':
            additional_type = row['additional_type(s)'].split(',')
            additional_data[row['primary_type']] = {
               'additional_type': additional_type
            }
   with open(community_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         if row['processed_labels'] != '':
            processed_labels = row['processed_labels'].split(';')
            additional_data[row['cell_type']] = {
               'processed_labels': processed_labels
            }
   with open(extraction_file, 'r', encoding='utf-8') as f:
      extraction_data = json.load(f)
   

   cnt1 = 0
   cnt2 = 0
   for item in extraction_data:
      cell_type = item['cell_type']
      item['additional_data'] = additional_data.get(cell_type, {}).get('additional_type', [])
      item['processed_labels'] = additional_data.get(cell_type, {}).get('processed_labels', [])
      if item['additional_data'] != []:
         cnt1 += 1
      if item['processed_labels'] != []:
         cnt2 += 1

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(extraction_data, f, indent=4, ensure_ascii=False)
   
def merge_with_flywire_using_additional_type(merged_alias_file: str, flywire_file: str, output_file: str):
   """Merge data with flywire CSV using both cell_type and aliases for matching."""
   # Read the merged data with aliases
   with open(merged_alias_file, 'r', encoding='utf-8') as f:
      merged_data = json.load(f)
   
   flywire_lookup = {}
   with open(flywire_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         cell_type = row['cell_type']
         flywire_lookup[cell_type] = {
               'root_id': row.get('root_id', ''),
               'flow': row.get('flow', ''),
               'super_class': row.get('super_class', ''),
               'class': row.get('class', ''),
               'sub_class': row.get('sub_class', ''),
               'hemibrain_type': row.get('hemibrain_type', ''),
               'hemilineage': row.get('hemilineage', ''),
               'side': row.get('side', ''),
               'nerve_feature': row.get('nerve_feature', ''),
               'found_in_flywire': True,
               'matched_by': 'direct'  
         }
   
   for item in merged_data:
      cell_type = item['cell_type']
      additional_type = item.get('additional_data', [])
      processed_labels = item.get('processed_labels', [])
      total_labels = additional_type + processed_labels
      found_in_flywire = item.get('found_in_flywire', False)
      
      if cell_type in flywire_lookup and not found_in_flywire:
         item.update(flywire_lookup[cell_type])
         continue
         
      if not found_in_flywire:
         for alias in total_labels:
            if alias in flywire_lookup:
               flywire_data = flywire_lookup[alias].copy()
               flywire_data['matched_by'] = 'community'  # Indicate match was through alias
               item.update(flywire_data)
               break
         else:
            item.update({
               'found_in_flywire': False,
               'matched_by': 'none'
            })

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(merged_data, f, indent=4, ensure_ascii=False)

def merge_with_neuprint(extraction_file, neuprint_file, output_file):
   """Merge extraction JSON with neuprint CSV using left join on cell_type.
   Matches if cell_type matches any of: type, subtype, systematicType, or hemilineage.
   Also matches with alias, additional_data, and processed_labels if available."""
   with open(extraction_file, 'r', encoding='utf-8') as f:
      extraction_data = json.load(f)
   
   neuprint_lookup = {} 
   with open(neuprint_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         # Create a set of all possible matching values
         matching_values = {
            row['type'],
            row['subtype'],
            row['systematicType'],
            row['hemilineage']
         }
         # Remove empty values
         matching_values = {v for v in matching_values if v}
         
         # For each matching value, create an entry in the lookup
         for value in matching_values:
            neuprint_lookup[value] = {
               'type': row['type'],
               'subtype': row['subtype'],
               'systematicType': row['systematicType'],
               'hemilineage': row['hemilineage'],
               'neuprint_id': row.get('bodyId', ''),
               'found_in_neuprint': True,
               'matched_by': 'neuprint',
               'type': row['type'],
               'subtype': row['subtype'],
               'systematicType': row['systematicType'],
               'hemilineage': row['hemilineage']
            }
   
   for item in extraction_data:
      cell_type = item['cell_type'] 
      found_in_flywire = item.get('found_in_flywire', False)
      found_in_neuprint = item.get('found_in_neuprint', False)
      
      # Get all possible matching values for this item
      all_possible_matches = {cell_type}
      
      # Add alias if exists
      if 'alias' in item:
         all_possible_matches.update(item['alias'])
      
      # Add additional_data if exists
      if 'additional_data' in item:
         all_possible_matches.update(item['additional_data'])
      
      # Add processed_labels if exists
      if 'processed_labels' in item:
         all_possible_matches.update(item['processed_labels'])
      
      # Try to find a match
      matched = False
      for match_value in all_possible_matches:
         if match_value in neuprint_lookup and not found_in_neuprint and not found_in_flywire:
            neuprint_data = neuprint_lookup[match_value].copy()
            if match_value != cell_type:
               neuprint_data['matched_by'] = 'alias' if match_value in item.get('alias', []) else \
                                           'additional_data' if match_value in item.get('additional_data', []) else \
                                           'processed_labels' if match_value in item.get('processed_labels', []) else \
                                           'direct'
            item.update(neuprint_data)
            matched = True
            break
      
      if not matched and not found_in_neuprint:
         item.update({
            'found_in_neuprint': False,
            'matched_by': 'none'
         })

   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(extraction_data, f, indent=4, ensure_ascii=False)

def main():
   parser = argparse.ArgumentParser(description='Merge cell type data from JSON and CSV files.')
   
   parser.add_argument('--input_folder', type=str, default='./python/assets/cell_types/gpt_4o',
                     help='Path to the folder containing JSON files')
   parser.add_argument('--extraction_file', type=str, default='./python/assets/merged_cell_types.json',
                     help='Path to save the merged extraction file')
   parser.add_argument('--flywire_file', type=str, default='./python/assets/flywire/classification.csv',
                     help='Path to the flywire CSV file')
   parser.add_argument('--output_file', type=str, default='./python/assets/merged_with_flywire.json',
                     help='Path to save the final merged file')
   parser.add_argument('--alias_file', type=str, default='./python/assets/alias_pairs.jsonl',
                     help='Path to save the alias pairs file')
   parser.add_argument('--alias_output', type=str, default='./python/assets/merged_with_alias.json',
                     help='Path to save the merged file with aliases')
   parser.add_argument('--final_output', type=str, default='./python/assets/merged_with_alias_with_flywire.json',
                     help='Path to save the final merged file with alias matching')
   args = parser.parse_args()
   
   # cell_types = merge_cell_types(args.input_folder, args.extraction_file)
   # merge_with_flywire(args.extraction_file, args.flywire_file, args.output_file)
   # statistics_report(args.output_file)
   # find_alias_pair(args.output_file, args.alias_file)
#  add_alias_to_merged_data(args.alias_file, args.output_file, args.alias_output)
   # merge_with_flywire_using_alias(args.alias_output, args.flywire_file, args.final_output)
   # statistics_report(args.final_output)  # Print statistics for the final merged file
   # merge_with_community_label("./python/assets/outputs/merged_with_alias_with_flywire.json", "./python/assets/flywire/consolidated_cell_types.csv", "./python/assets/flywire/merged_labels.csv", "./python/assets/outputs/merged_with_alias_and_community.json")
   # merge_with_flywire_using_additional_type("./python/assets/outputs/merged_with_alias_and_community.json", "./python/assets/flywire/classification.csv", "./python/assets/outputs/merged_with_alias_and_community_and_flywire.json")
   merge_with_neuprint("./python/assets/outputs/merged_with_alias_and_community_and_flywire.json", "./python/assets/neuprint/all_neurons_data.csv", "./python/assets/outputs/merged_with_alias_and_community_and_flywire_and_neuprint.json")

if __name__ == "__main__":
    main()