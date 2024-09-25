import gzip
import os.path
import pickle
import random

from sentence_transformers import util
from torch.utils.data import DataLoader, DistributedSampler

from sign2sem.tools.common import get_iter_gzip
from sign2sem.tools.datasets import SignDataset


def get_data_loaders(cfg, rank, device, world_size, get_test=False):
    train_pth = os.path.join(cfg["root_path"] if "root_path" in cfg else "", cfg["train_path"])
    dev_pth = os.path.join(cfg["root_path"] if "root_path" in cfg else "", cfg["dev_path"])
    train_data = SignDataset(train_pth, aug=cfg["aug"] if "aug" in cfg else 1)
    # Define your train dataset, the dataloader and the train loss
    distributed_sampler = DistributedSampler(dataset=train_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_data, batch_size=cfg["train_batch_size"],
                                  sampler=distributed_sampler)
    dev = get_iter_gzip(dev_pth)
    signs1 = []
    signs2 = []
    scores = []
    for i in dev:
        j = random.choice(dev)
        signs1.append(i["sign"].to(device))
        signs2.append(j["sign"].to(device))
        embeddings = train_data.model.encode([i["text"], j["text"]], convert_to_tensor=True)
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        scores.append(score.detach().cpu().item())
    dev_data_loader = {"sign1": signs1, "sign2": signs2, "scores": scores}
    return train_dataloader, dev_data_loader
