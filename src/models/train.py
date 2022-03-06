import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from src.models.CellPaintingDataset import CellPaintingDataset
from src.models.CellPaintingModel import CNN
from src.models.config import Config

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_stratified_sampler(config, generator):
    df = pd.read_csv(config.dataset_metadata)
    cartridges_names = list(df.folder.unique())
    test_cartridge = cartridges_names.pop(np.random.randint(0, len(cartridges_names)))
    test_index = df[df.folder.str.match(test_cartridge)].index.to_numpy()
    train_df = df[~df.index.isin(test_index)]
    y = train_df[['compound_label']].to_numpy()
    x = np.array(list(range(len(y)))).reshape(-1, 1)
    g = train_df[['well_id']].to_numpy()
    sgs = StratifiedGroupKFold(n_splits=8, random_state=0, shuffle=True)
    (train_index, val_index) = list(sgs.split(x, y, g))[0]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=generator)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=generator)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=generator)
    return train_sampler, val_sampler, test_sampler


def prepare_data(config, data_transform):
    dataset = CellPaintingDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)
    g = torch.Generator(device=device).manual_seed(0)
    train_sampler, val_sampler, test_sampler = get_stratified_sampler(config, g)
    loader_params = dict(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=config.pin_memory, generator=g)
    train_loader = torch.utils.data.DataLoader(**loader_params, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(**loader_params, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(**loader_params, sampler=test_sampler)
    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


def prepare_model(model_architecture, config):
    model = CNN(model_architecture, num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    return model, criterion, optimizer, exp_lr_scheduler


def train_log(loss, accuracy, example_ct, epoch, is_train):
    if is_train:
        wandb.log({"epoch": epoch, "loss": loss, "acc": accuracy * 100.}, step=example_ct)
    else:
        wandb.log({"epoch": epoch, "val_loss": loss, "val_acc": accuracy * 100.}, step=example_ct)


def train(model, criterion, optimizer, scheduler, dataloaders, config):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, example_ct, batch_ct = 0.0, 0, 0
    for epoch in tqdm(range(config.epochs)):
        for phase in ['train', 'val']:
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0
            num_examples = 0
            with tqdm(dataloaders[phase], unit="batch", leave=False) as tepoch:
                for _, (inputs, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    labels = labels.to(device)
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    example_ct += len(inputs)
                    batch_ct += 1
                    correct = torch.sum(predictions == labels.data).detach().cpu().numpy()
                    accuracy = correct / config.batch_size
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += correct
                    num_examples += inputs.shape[0]
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    # train_log(loss, accuracy, example_ct, epoch, phase == 'train')
            if phase == 'train':
                scheduler.step()
            epoch_acc = running_corrects / num_examples
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    config = Config()
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[163.24, 536.39, 425.26, 581.64],
                             std=[204.27, 1386.95, 917.2, 519.7])
    ])

    dataloaders = prepare_data(config, data_transform)

    try:
        model = torch.load(config.model_path)
        model.eval()
    except FileNotFoundError:
        model = models.resnet18(pretrained=True)
        model, criterion, optimizer, scheduler = prepare_model(model, config)
        # with wandb.init(project="CellPainting", config=config.__dict__):
        model = train(model, criterion, optimizer, scheduler, dataloaders, config)
        torch.save(model, config.model_path)