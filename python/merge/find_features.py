import json
from feature_prompt import SYSTEM_PROMPT_EXTRACT_FEATURES, FEW_SHOT_EXTRACT_FEATURES
import tqdm
import argparse
import pandas as pd

def complete_missing_features(input_file: str, output_file: str):
   """Complete missing features for neurons not found in flywire using LLM."""
   model_type = "gpt-4o"
   model_kwargs = {"response_format": {"type": "json_object"}}
   default_config = {"temperature": 0, "frequency_penalty": 0.1}
   from langchain_openai import ChatOpenAI
   from langchain.prompts import ChatPromptTemplate

   # Create model and prompt
   model = ChatOpenAI(
      model=model_type,
      model_kwargs=model_kwargs,
      **default_config,
   )
   prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT_EXTRACT_FEATURES + "\n\n" + FEW_SHOT_EXTRACT_FEATURES),
      ("user", "{input}")
   ])
   chain = prompt | model

   # Read the input file
   with open(input_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
   
   # Process each item not found in flywire
   for item in tqdm.tqdm(data):
      if not item.get('found_in_flywire', False):
         # Collect all descriptions
         descriptions = []
         for extraction in item['extractions']:
            descriptions.append(extraction['explanation'])
            descriptions.extend(extraction['factoids'])
         
         # Filter out empty descriptions and strip whitespace
         descriptions = [d.strip() for d in descriptions if d.strip()]
         
         # Prepare input for the model
         input_data = {
            "cell_type": item['cell_type'],
            "descriptions": descriptions
         }
         
         try:
            response = chain.invoke({"input": json.dumps(input_data) + "\n\nPlease respond in JSON format."}).content
            features = json.loads(response)
            
            # Update the item with predicted features
            item.update({
               'flow': features.get('flow', 'None'),
               'superclass': features.get('superclass', 'None'),
               'class': features.get('class', 'None'),
               'features_source': 'llm'
            })
            
            print(f"\nProcessed {item['cell_type']}:")
            print(f"Flow: {features.get('flow', 'None')}")
            print(f"Superclass: {features.get('superclass', 'None')}")
            print(f"Class: {features.get('class', 'None')}\n")
               
         except Exception as e:
            print(f"Error processing {item['cell_type']}: {str(e)}")
            item.update({
               'flow': 'None',
               'superclass': 'None',
               'class': 'None',
               'features_source': 'error'
            })
         
         # Save the results
         with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


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


def match_by_features(fly_wire_path: str, input_file: str, output_file: str):
   """Match neurons not found in flywire with potential candidates based on features."""
   flywire_df = pd.read_csv(fly_wire_path)

   with open(input_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
   
   for item in data:
      if not item.get('found_in_flywire', False):
         flow = item.get('flow', 'None')
         superclass = item.get('superclass', 'None')
         class_ = item.get('class', 'None')

         mask = pd.Series(True, index=flywire_df.index)

         # Match flow - including None
         if flow == 'None':
            mask &= (flywire_df['flow'].isna() | (flywire_df['flow'] == ''))
         else:
            mask &= (flywire_df['flow'] == flow)
         
         # Match superclass - including None
         if superclass == 'None':
            mask &= (flywire_df['super_class'].isna() | (flywire_df['super_class'] == ''))
         else:
            mask &= (flywire_df['super_class'] == superclass)

         # Match class - including None
         if class_ == 'None':
            mask &= (flywire_df['class'].isna() | (flywire_df['class'] == ''))
         else:
            mask &= (flywire_df['class'] == class_)

         candidates = flywire_df[mask].to_dict('records')
         candidates_cells = [candidate['cell_type'] for candidate in candidates]
         
         item['matching_candidates'] = candidates_cells
         
         print(f"\nMatching candidates for {item['cell_type']}:")
         print(f"Flow: {flow}, Superclass: {superclass}, Class: {class_}")
         print(f"Found {len(candidates)} candidates")
         # for candidate in candidates:
         #       print(f"- {candidate['cell_type']} (Flow: {candidate['flow']}, Superclass: {candidate['super_class']}, Class: {candidate['class']})")
   # Save the results
   with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Complete missing features for neurons not found in flywire using LLM.')
   parser.add_argument('--input_file', type=str, default='./python/assets/merged_with_alias_with_flywire.json',
                     help='Path to the input file')
   parser.add_argument('--features_file', type=str, default='./python/assets/merged_with_alias_with_flywire_features.json',
                     help='Path to save the output file')
   parser.add_argument('--match_file', type=str, default='./python/assets/merged_with_alias_flywire_feature_match.json',
                     help='Path to save the output file')
   parser.add_argument('--flywire_path', type=str, default='./python/assets/flywire/classification.csv',
                     help='Path to the flywire data')
   args = parser.parse_args()
   # complete_missing_features(args.input_file, args.features_file)
   match_by_features(args.flywire_path, args.features_file, args.match_file)
