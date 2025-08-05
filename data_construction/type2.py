import pandas as pd
import json
import random


selfies_file_path = 'selfies.csv'  # change it to fit your workflow
selfies_data = pd.read_csv(selfies_file_path)


qa_data = []


def mask_selfies_continuous(selfies, fraction):
    fragments = selfies.split('][')
    fragments[0] = fragments[0] + ']'  
    fragments[-1] = '[' + fragments[-1]  
    fragments[1:-1] = ['[' + frag + ']' for frag in fragments[1:-1]]  
    

    total_fragments = len(fragments)
    mask_length = max(1, int(total_fragments * fraction))
    

    start_index = random.randint(0, total_fragments - mask_length)
    end_index = start_index + mask_length
    

    masked_selfies = fragments[:start_index] + ['[MASK]'] + fragments[end_index:]
    
    return ''.join(masked_selfies)


for _, row in selfies_data.iterrows():
    original_selfies = row['selfies']  


    for fraction in [1/3, 1/5]:
        masked_selfies = mask_selfies_continuous(original_selfies, fraction)
        qa_entry = {
            "instruction": "Predict the missing molecular fragments in this ligand's representation using your chemical expertise. You are an expert in ligand prediction, and [MASK] indicates the parts you need to infer. The predicted fragments can vary in length.",
            "input": masked_selfies,
            "output": original_selfies
        }
        qa_data.append(qa_entry)


output_file_path = 'continuous_masked_ligand_qa_dataset.json'  
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(qa_data, json_file, ensure_ascii=False, indent=2)

print(f"type2 dataset is saved, {output_file_path}")
