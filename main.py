import torch
import clip
from PIL import Image
from os import listdir
import os
from os.path import isfile, join
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
device = "cuda" if torch.cuda.is_available() else "cpu"
models_names = ['ViT-B/32']#, 'RN50', 'RN101', 'RN50x4']
models = {}
preprocesses = {}
for name in models_names:
    models[name], preprocesses[name] = clip.load(name, device=device)

print(clip.available_models())
# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def predict(image):
    for name in models_names:
        image = preprocesses[name](Image.open(image)).unsqueeze(0).to(device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        with torch.no_grad():
            image_features = models[name].encode_image(image)
            text_features = models[name].encode_text(text_inputs)
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

            # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print("\n\n"+str(name))
            print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        """
        plt.imshow(image)
        plt.show()
        """
mypath = "./examples/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
    predict(join(mypath,file))

