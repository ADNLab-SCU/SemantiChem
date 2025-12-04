import pandas as pd
import json
import random


selfies_file_path = 'selfies.csv'  # change it to fit your workflow
selfies_data = pd.read_csv(selfies_file_path)

# instructions
instructions = [
    "Can you provide an example of an existing ligand?",
    "Show me an example of a known ligand molecule.",
    "Please give me a molecular representation of an existing ligand."
]

qa_data = []

for _, row in selfies_data.iterrows():
    molecule = row['selfies']  
    qa_entry = {
        "instruction": random.choice(instructions), 
        "input": "",
        "output": molecule 
    }
    qa_data.append(qa_entry)


output_file_path = 'ligand_qa_dataset.json'  
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(qa_data, json_file, ensure_ascii=False, indent=2)

print(f"type1 dataset is saved, {output_file_path}")
