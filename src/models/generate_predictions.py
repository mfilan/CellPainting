import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tifffile import imread

from src.models import train
from src.models.config import Config

config = Config()
g = torch.Generator(device=train.device).manual_seed(0)
training, validation, testing = train.get_stratified_sampler(config, g)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[163.24, 536.39, 425.26, 581.64],
                         std=[204.27, 1386.95, 917.2, 519.7])
])

class_mapping = {
    0: "Berberine Chloride",
    2: "DFSO",
    1: "Brefeldin A",
    3: "Fluphenazine",
    4: "Latrunculin B",
    5: "Nocodazole",
    6: "Rapamycin",
    7: "Rotenone",
    8: "Tetrandrine"
}

results = []

for i, row in enumerate(pd.read_csv(config.dataset_metadata).itertuples()):
    image = imread(os.path.join(config.data_root_dir, row.filename)).astype(np.float32)
    image = data_transform(image)
    image = torch.FloatTensor(np.expand_dims(image, axis=0))
    output = train.model(image)
    results.append({
        'image_name': f"{row.folder}/{row.filename[:-7]}",
        'dataset': "training" if i in training else "validation" if i in validation else "testing",
        'true_compound': row.compound_name,
        'predicted_compound': class_mapping[(torch.max(output, 1)[1]).item()],
        'concentration': row.concentration_name,
        **{
            class_mapping[i]: p.item()
            for i, p in enumerate(output[0].detach().cpu())
        }
    })

pd.DataFrame(results).to_csv(config.model_predictions, index=False)