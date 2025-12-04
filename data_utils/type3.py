import pandas as pd
import json
import random


selfies_file_path = 'selfies.csv'  # change it to fit your workflow
selfies_data = pd.read_csv(selfies_file_path)


qa_data = []


def mask_selfies_scattered(selfies, fraction):
    fragments = selfies.split('][')
    if not fragments:
        return selfies

    fragments[0] = fragments[0] + ']'
    fragments[-1] = '[' + fragments[-1]
    

    fragments[1:-1] = ['[' + frag + ']' for frag in fragments[1:-1]]
    
    total = len(fragments)
    mask_count = max(1, round(total * fraction))  
    mask_count = min(mask_count, total) 
    

    masked_indices = random.sample(range(total), mask_count)
    

    masked_fragments = [
        '[SINGLEMASK]' if i in masked_indices else frag 
        for i, frag in enumerate(fragments)
    ]
    
    return ''.join(masked_fragments)


for _, row in selfies_data.iterrows():
    original_selfies = row['selfies']  


    for fraction in [1/3, 1/5]:
        masked_selfies = mask_selfies_scattered(original_selfies, fraction)
        qa_entry = {
            "instruction": "As a ligand prediction expert, use your chemical knowledge to predict the molecular fragments hidden behind [SINGLESINGLEMASK] in this ligand's structure. Note that every [SINGLEMASK] is a fragment of length one.",
            "input": masked_selfies,
            "output": original_selfies
        }
        qa_data.append(qa_entry)


output_file_path = 'scattered_masked_ligand_qa_dataset.json'
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(qa_data, json_file, ensure_ascii=False, indent=2)

print(f"type3 dataset is saved, {output_file_path}")
