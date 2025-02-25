import torch
import numpy as np


if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")
    
repo = 'OxWearables/ssl-wearables'
harnet5 = torch.hub.load(repo, 'harnet5', class_num=8, pretrained=True)
x = np.random.rand(1, 3, 150)
x = torch.FloatTensor(x)
print(harnet5(x))

harnet10 = torch.hub.load(repo, 'harnet10', class_num=8, pretrained=True)
x = np.random.rand(1, 3, 300)
x = torch.FloatTensor(x)
print(harnet10(x))

harnet30 = torch.hub.load(repo, 'harnet30', class_num=8, pretrained=True)
x = np.random.rand(1, 3, 900)
x = torch.FloatTensor(x)
print(harnet30(x))


# read npy data

Y = np.load('/data/ssl_wearable/data/downstream/realworld_30hz_clean/Y.npy')
# find the unique labels
unique_labels = np.unique(Y)
# count the number of unique labels per class sample
pivot = np.zeros(len(unique_labels))
for i, label in enumerate(unique_labels):
    pivot[i] = np.sum(Y == label)
# print the unique labels and their counts
print(unique_labels)
print(pivot)
