import random

import sentence_transformers
import torch
from sentence_transformers import util
from torch.utils import data

from sign2sem.models.sem_transformer import SignSample
from sign2sem.tools.common import get_iter_gzip


class SignDataset(data.Dataset):
    def __init__(self, npy_pth, model_name="paraphrase-multilingual-MiniLM-L12-v2", base_prob=0.05, aug=10):
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.data = get_iter_gzip(npy_pth)
        # self.data += get_iter_gzip(npy_pth.replace("train", "dev"))
        # self.data += get_iter_gzip(npy_pth.replace("train", "test"))

        self.base_prob = base_prob
        self.aug = aug

    def __len__(self):
        return len(self.data) * self.aug

    def __getitem__(self, index):
        a = self.data[index % len(self.data)]
        b = random.choice(self.data)
        ret = [a["text"], b["text"]]
        random_mask_a = torch.rand(a["sign"].size(0)) > (self.base_prob * (index // len(self.data)))
        random_mask_b = torch.rand(b["sign"].size(0)) > (self.base_prob * (index // len(self.data)))
        embeddings = self.model.encode(ret, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return SignSample(signs=[a["sign"] * random_mask_a.unsqueeze(-1),
                                 b["sign"] * random_mask_b.unsqueeze(-1)], label=score)
