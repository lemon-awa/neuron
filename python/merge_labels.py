import pandas as pd
import ast

# Read the CSV files
classification_df = pd.read_csv('assets/flywire/classification.csv')
processed_labels_df = pd.read_csv('assets/flywire/processed_labels.csv')

# Process the processed_labels column
def process_labels(label_str):
    try:
        # Convert string representation of list to actual list
        label_list = ast.literal_eval(label_str)
        # Join all elements with semicolon
        return '; '.join(label_list)
    except:
        return label_str

processed_labels_df['processed_labels'] = processed_labels_df['processed_labels'].apply(process_labels)

# Merge the dataframes on root_id
merged_df = pd.merge(classification_df, processed_labels_df, on='root_id', how='outer')

# Save the merged result
merged_df.to_csv('assets/flywire/merged_labels.csv', index=False)

print("Files merged successfully!")
print(f"Total rows in merged file: {len(merged_df)}") 