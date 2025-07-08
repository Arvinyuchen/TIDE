import pprint

def whitespace_tokenizer(prompt: str):
    tokens = prompt.strip().split()
    word_to_indices = {word: [i] for i, word in enumerate(tokens)}
    return tokens, word_to_indices
