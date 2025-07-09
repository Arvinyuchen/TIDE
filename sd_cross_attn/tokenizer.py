import torch
from pipeline_attend_and_excite import AttendAndExcitePipeline
from typing import List, Dict
from transformers import CLIPTokenizer



def get_token_indices(prompt, concepts, tokenizer):
    tokenized = tokenizer(prompt, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
    print(tokens)

    def norm(token):
        return token.lower().replace('</w>', '')
    norm_prompt = [norm(t) for t in tokens]

    print(norm_prompt)

    concept_indices: Dict[str, List[int]] = {}
    for concept in concepts:
        concept_ids = tokenizer(concept, add_special_tokens=False).input_ids
        concept_tokens = tokenizer.convert_ids_to_tokens(concept_ids)
        normalized_concept = ''.join([norm(t) for t in concept_tokens])

        for i in range(len(norm_prompt) - len(concept_tokens) + 1):
            window = ''.join(norm_prompt[i:i + len(concept_tokens)])
            if window == normalized_concept:
                concept_indices[concept] = list(range(i, i + len(concept_tokens)))
                break  # match first occurrence
    
    return concept_indices



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
stable = AttendAndExcitePipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
tokenizer = stable.tokenizer

prompt   = "a photo of a dog with floppy ears and whiskers"
concepts = ["floppy ears", "whiskers", "tail"]

idx_dict = get_token_indices(prompt, concepts, tokenizer)
print(idx_dict)