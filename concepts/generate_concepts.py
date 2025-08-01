import openai
import json
import time
from pathlib import Path

# Fill the OpenAI API key
openai.api_key = "PLEASE FILL YOUR OWN API KEY"

# Get all the class names(categories) from the dataset
def get_class_names_by_dataset(dataset_name):
    if dataset_name.lower() == "pacs":
        class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

    elif dataset_name.lower() == "vlcs":
        class_names = ["bird", "car", "chair", "dog", "person"]

    elif dataset_name.lower() == "officehome":
        folder_path = Path(f"./data/OfficeHome/art/train")
        class_names = sorted([f.name for f in folder_path.iterdir() if f.is_dir()])

    elif dataset_name.lower() == "domainnet":
        txt_file_path = 'data/DomainNet/split/clipart_test_mini.txt'
        class_names = set()
        with open(txt_file_path, 'r') as f:
            for line in f:
                path = line.strip().split()[0]
                class_name = path.split('/')[1]
                class_names.add(class_name)
        class_names = sorted(list(class_names))

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    return class_names


# Use GPT-3.5 to generate a list of concepts for a given class name
def generate_concepts_for_class(class_name):
    # Different from the original prompt in TIDE paper.
    # The last sentence is added to condition the generation, otherwise
    # the outputs are not usable
    prompt = f"""List the most visually distinctive and static features of a {class_name} that a classification model would rely on for accurate identification. Focus only on domain-agnostic features that are intrinsic to the object itself and truly discriminative for the class, avoiding any features that may be related to the environment or context in which the object is typically found. Each concept should only described in one word.

    Example:
    Input: cat
    Output: whiskers, eyes, ears
    
    Input: {class_name}
    Output:
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            # Adjusting the degree of deterministic
            temperature=0.2
        )

        concepts = response.choices[0].message.content
        concept_list = [c.strip() for c in concepts.split(",")]
        return concept_list
    
    except Exception as e:
        print(f"Error for class {class_name}: {e}")
        return []
    

# A dictionary mappin each class name to a list of concepts generated by GPT-3.5
def build_concept_dict(class_names):
    concept_dict = {}
    for class_name in class_names:
        concepts = generate_concepts_for_class(class_name)
        print(f"Concepts for {class_name}: {concepts}")
        concept_dict[class_name] = concepts
        time.sleep(2)
    
    return concept_dict



if __name__ == "__main__":
    datasets = ["pacs", "vlcs", "officehome", "domainnet"]
    for dataset_name in datasets:
        class_names = get_class_names_by_dataset(dataset_name)
        concept_dict = build_concept_dict(class_names)

        # Save to JSON file
        with open(f"concepts/{dataset_name}_concepts.json", "w") as f:
            json.dump(concept_dict, f, indent=4)

        print("\nConcepts generation completed! Results saved to concepts directory")

