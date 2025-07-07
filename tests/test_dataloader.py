from tide.data.datasets import get_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Test with download enabled
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# Change the first argument str to name the dataset, and the root accordingly
ds = get_dataset(
    "domainnet",
    root="./data/DomainNet", 
    split="train", 
    domain="clipart", 
    transform=transform
)

print(f"Dataset size: {len(ds)} samples")
img, label, domain_id = ds[0]
print(f"Label: {label}, Domain ID: {domain_id}")

# Plot the image
plt.imshow(F.to_pil_image(img))  # Convert Tensor to PIL image for display
plt.title(f"Label: {label}, Domain ID: {domain_id}")
plt.axis('off')
plt.show()