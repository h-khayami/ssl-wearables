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


# import os
# import pandas as pd
# save_path = os.path.expanduser('~/ssl-wearables/data/reports/2025-02-12_18-13/p1.csv')  # Expand ~
# parent_dir = os.path.dirname('~/ssl-wearables/data/reports/2025-02-12_18-13/p1.csv')  # Extract parent directory

# if not os.path.exists(parent_dir):
#     print(f"Creating missing directory: {parent_dir}")
#     os.makedirs(parent_dir, exist_ok=True)
# df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
# df.to_csv(save_path, index=False)