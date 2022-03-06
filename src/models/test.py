from train import *

config = Config()
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[163.24, 536.39, 425.26, 581.64],
                         std=[204.27, 1386.95, 917.2, 519.7])
])

dataloaders = prepare_data(config, data_transform)
print(list(dataloaders['train']))