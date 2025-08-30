import json
import random
import re
from objprint import op
from tqdm import tqdm

from api import invoke_api
    
    
if __name__ == '__main__':
    input = '../data/nr3d_val.json'
    with open(input, 'r') as f:
        prog_data = json.load(f)
    
    with open('../prompts/prompt_obj.txt', 'r') as f:
        system_prompt = f.read()
    
    idx = 0
    new_data = []
    prog_data = prog_data[:500]
    
    pbar = tqdm(total=len(prog_data))
    while idx < len(prog_data):
        batch_idx = 0
        user_prompt = 'Please follow the format of examples strictly.\n'
        while batch_idx < 10:
            caption = prog_data[idx]['caption']
            user_prompt += f"[{idx}] Query: {caption}\n"
            idx += 1
            batch_idx += 1
            pbar.update(1)        
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        vlm_output = invoke_api('doubao', messages)
        answers = vlm_output['answer'].split('\n')
        
        for answer in answers:
            pattern = r'\[(\d+)\]'
            match = re.search(pattern, answer)

            if not match:
                op(vlm_output)
                assert False
            
            data_dict = prog_data[int(match.group(1))]
            obj_name = answer.split(':')[1].strip()
            data_dict['obj_name'] = obj_name
            new_data.append(data_dict)
        
    json.dump(new_data, open('../data/nr3d.json', 'w'), indent=4)
