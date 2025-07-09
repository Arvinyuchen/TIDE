import os
import math
from typing import List, Dict
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch

import ptp_utils
from ptp_utils import AttentionStore, aggregate_attention



def get_token_indices(prompt, concepts, tokenizer):
    tokenized = tokenizer(prompt, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])

    indices = []
    for concept in concepts:
        concept_tokens = tokenizer(concept, add_special_tokens=False).input_ids
        for i in range(len(tokens) - len(concept_tokens) + 1):
            if tokenized.input_ids[0][i:i + len(concept_tokens)].tolist() == concept_tokens:
                indices.extend(list(range(i, i + len(concept_tokens))))
                break
    return indices




def get_conccept_indices(prompt, concepts, tokenizer):
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



# def show_cross_attention(prompt: str,
#                          attention_store: AttentionStore,
#                          tokenizer,
#                          class_name: str,
#                          indices_to_alter: List[int],
#                          res: int,
#                          from_where: List[str],
#                          select: int = 0,
#                          orig_image=None):
#     output_dir = os.path.join("outputs", class_name)
#     os.makedirs(output_dir, exist_ok=True)

#     tokens = tokenizer.encode(prompt)
#     decoder = tokenizer.decode
 
#     attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
#     images = []

#     # show spatial attention for indices of tokens to strengthen
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         if i in indices_to_alter:
#             image = show_image_relevance(image, orig_image)
#             image = image.astype(np.uint8)
#             image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
#             token_text = decoder(int(tokens[i]))

#             image = ptp_utils.text_under_image(image, token_text)
#             images.append(image)

#             output_path = os.path.join(output_dir, f"{token_text}_attention.png")
#             Image.fromarray(image).save(output_path)

#     ptp_utils.view_images(np.stack(images, axis=0))



# def show_cross_attention(prompt: str,
#                          attention_store: AttentionStore,
#                          tokenizer,
#                          class_name: str,
#                          concepts: list,
#                          res: int,
#                          from_where: list,
#                          select: int = 0,
#                          orig_image=None):
#     output_dir = os.path.join("outputs", class_name)
#     os.makedirs(output_dir, exist_ok=True)

#     token_groups = get_conccept_indices(prompt, concepts, tokenizer)

#     # Aggregate attention maps over different layers and heads
#     attention_maps = aggregate_attention(attention_store, res, from_where, is_cross=True, select=select).detach().cpu()

#     all_images = []
#     for concept, indices in token_groups.items():
#         relevance = None
#         for idx in indices:
#             image = attention_maps[:, :, idx]
#             if orig_image is not None:
#                 image = show_image_relevance(image, orig_image)
#             if relevance is None:
#                 relevance = image
#             else:
#                 relevance = image

#         relevance = np.clip(relevance / len(indices), 0, 255).astype(np.uint8)
#         image_resized = np.array(Image.fromarray(relevance).resize((res**2, res**2)))
#         image_labeled = ptp_utils.text_under_image(image_resized, concept)
#         output_path = os.path.join(output_dir, f"{class_name}_{concept}_attention.png")
#         Image.fromarray(image_labeled).save(output_path)
#         all_images.append(image_labeled)

#     if all_images:
#         ptp_utils.view_images(np.stack(all_images, axis=0))




def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         class_name: str,
                         concepts: list,
                         res: int,
                         from_where: list,
                         select: int = 0,
                         orig_image=None,
                         debug_show_raw=False):
    import matplotlib.pyplot as plt

    output_dir = os.path.join("outputs", class_name)
    os.makedirs(output_dir, exist_ok=True)

    token_groups = get_conccept_indices(prompt, concepts, tokenizer)

    # Aggregate attention maps over layers/heads
    attention_maps = aggregate_attention(attention_store, res, from_where, is_cross=True, select=select).detach().cpu()

    all_images = []
    for concept, indices in token_groups.items():
        if not indices:
            continue  # skip if concept had no matching tokens

        relevance = torch.zeros_like(attention_maps[:, :, 0])
        for idx in indices:
            relevance += attention_maps[:, :, idx]

        # Normalize raw relevance before visualization
        relevance /= len(indices)
        relevance_np = relevance.numpy()

        # Optional debug: show raw attention map before overlay
        if debug_show_raw:
            raw_vis = (relevance_np - relevance_np.min()) / (relevance_np.max() - relevance_np.min() + 1e-8)
            raw_vis = (raw_vis * 255).astype(np.uint8)
            raw_vis = cv2.applyColorMap(raw_vis, cv2.COLORMAP_JET)
            plt.imshow(raw_vis[..., ::-1])
            plt.title(f"Raw Attention: {concept}")
            plt.axis("off")
            plt.show()

        # Overlay with original image
        if orig_image is not None:
            vis = show_image_relevance(relevance, orig_image)
        else:
            vis = relevance_np  # fallback

        vis_resized = np.array(Image.fromarray(vis).resize((res ** 2, res ** 2)))
        labeled_img = ptp_utils.text_under_image(vis_resized, concept)
        output_path = os.path.join(output_dir, f"{class_name}_{concept}_attention.png")
        Image.fromarray(labeled_img).save(output_path)
        all_images.append(labeled_img)

    if all_images:
        ptp_utils.view_images(np.stack(all_images, axis=0))




def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image