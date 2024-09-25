import os

import torch
from sentence_transformers.evaluation import SimilarityFunction
from torch.nn.parallel import DistributedDataParallel
from transformers import BertConfig

from sign2sem.models.sem_transformer import SentenceTransformer, SignBertModel, EmbeddingSimilarityEvaluatorSign, \
    CosineSimilarityLoss
from sign2sem.tools.data_utils import get_data_loaders


# Define the model. Either from scratch of by loading a pre-trained model
def create_sign_sem_model(base="paraphrase-MiniLM-L3-v2", input_size=1024):
    # TODO: extend to all sBERT
    sem_transformers = {"paraphrase-MiniLM-L3-v2": "nreimers/MiniLM-L3-H384-uncased",
                        "paraphrase-multilingual-MiniLM-L12-v2": "microsoft/Multilingual-MiniLM-L12-H384"}
    model = SentenceTransformer(base)
    config = BertConfig.from_pretrained(sem_transformers[base])
    bertS = SignBertModel(config, input_size=input_size)
    bertS.copy_from_bert(model[0].auto_model)
    bertS.encoder = model[0].auto_model.encoder
    bertS.pooler = model[0].auto_model.pooler
    model[0].auto_model = bertS
    return model


def freeze_topk(model, n=2):
    for name, param in model.named_parameters():
        if "encoder.layer" in name and "attention" in name:
            # Freeze all attention layers except layer 0
            layer_number = int(name.split(".")[4])
            if layer_number > n:
                param.requires_grad = False
        elif "layer_norm" in name:
            # Keep all layer norms trainable
            param.requires_grad = True
        elif "embeddings" in name:
            # Keep the entire embeddings layer trainable
            param.requires_grad = True
        elif "pooler" in name:
            param.requires_grad = True
        else:
            # Freeze other layers
            param.requires_grad = False


def train(cfg):
    torch.distributed.init_process_group(backend='nccl')

    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    world_size = torch.distributed.get_world_size()

    # Your model, tokenizer, dataset, and other setup code
    model = create_sign_sem_model(cfg["model_base_name"], cfg["input_size"]).cuda(local_rank)
    # Wrap model with DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    train_loader, dev_loader = get_data_loaders(cfg, local_rank, device, world_size)

    evaluator = EmbeddingSimilarityEvaluatorSign(dev_loader["sign1"], dev_loader["sign2"], dev_loader["scores"],
                                                 write_csv=True,
                                                 main_similarity=SimilarityFunction.COSINE)

    print(f"steps;epoch;score\n",
          file=open(os.path.join(cfg["ckpt_path"], "history_best.csv"), mode="w", encoding="utf-8"))
    train_loss = CosineSimilarityLoss(model.module)
    # freeze_topk(model)
    model.module.fit(train_objectives=[(train_loader, train_loss)], epochs=cfg["n_epochs"], warmup_steps=cfg["warmup"],
                     evaluation_steps=cfg["eval_step"], callback=None, output_path=cfg["ckpt_path"],
                     checkpoint_path=cfg["ckpt_path"], evaluator=evaluator,
                     checkpoint_save_total_limit=cfg["ckpt_limit"], optimizer_params={"lr": float(cfg["lr"])},
                     scheduler=cfg["scheduler"])
