import abc
import torch

from diffusers import StableDiffusionPipeline

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0 
        
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0



class AttentionStore(AttentionControl):

    def __init__(self):
        super().__init__()
        self.attention_maps = {}

    def forward(self, attn, is_corss, place_in_unet):
        if is_corss:
            if place_in_unet not in self.attention_maps:
                self.attention_maps[place_in_unet] = []
            self.attention_maps[place_in_unet].append(attn.detach().cpu())

def register_attention_control(unet, controller):
    def attn_hook(module, input, output):
        controller(module, is_cross=True, place_in_unet='down')
    
    for name, module in unet.named_modules():
        if "attn2" in name:
            module.register_forward_hook(attn_hook)



def run_sd_with_attention(prompt, controller, model_id="stabilityai/stable-diffusion-2-1"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    register_attention_control(pipe.unet, controller)

    image = pipe(prompt=prompt, num_inference_steps=50).images[0]
    return image, controller.attention_maps


if __name__ == "__main__":
    # Example usage
    from PIL import Image
    import matplotlib.pyplot as plt

    prompt = "a dog with fur, tail, paws"
    controller = AttentionStore()

    image, attention_maps = run_sd_with_attention(prompt, controller)

    image.save("generated_image.png")
    print("Image generated and saved!")

    # Example to display one attention map
    for layer_name, maps in attention_maps.items():
        print(f"Layer: {layer_name}, Attention map shape: {maps[0].shape}")
        break  # Show one for now

    plt.imshow(image)
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()