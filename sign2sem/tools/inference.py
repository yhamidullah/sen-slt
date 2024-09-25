import argparse
import glob
import gzip
import pickle

from tqdm import tqdm
from transformers import BertConfig
from sign2sem.models.sem_transformer import SentenceTransformer, SignBertModel


def create_sign_sem_model(base="paraphrase-MiniLM-L3-v2", input_size=1024):
    sem_transformers = {"paraphrase-MiniLM-L3-v2": "nreimers/MiniLM-L3-H384-uncased",
                        "paraphrase-multilingual-MiniLM-L12-v2": "microsoft/Multilingual-MiniLM-L12-H384"}
    model = SentenceTransformer(base)
    print(f"model params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    config = BertConfig.from_pretrained(sem_transformers[base])
    bertS = SignBertModel(config, input_size=input_size)
    print(
        f"model params : {sum(p.numel() for p in model[0].auto_model.embeddings.word_embeddings.parameters() if p.requires_grad)}")
    bertS.copy_from_bert(model[0].auto_model)

    model[0].auto_model = bertS
    return model


def get_iter_gzip(sgn_path):
    f = gzip.open(sgn_path, "rb")
    folders = pickle.load(f)
    return folders


parser = argparse.ArgumentParser(description='Inference of Sign2Sem')
parser.add_argument('--ckpt', type=str, help='config path', required=True)
parser.add_argument('--in_path', type=str, help='config path', required=True)
parser.add_argument('--out_path', type=str, help='config path', required=True)
parser.add_argument('--d_sem', type=int, default=32, help="d_sem : split*d_sem=sem_dim")
parser.add_argument('--sem_dim', type=int, default=384, help="sem_dim : split*d_sem=sem_dim")
parser.add_argument('--input_size', type=int, default=1024, help="feature input size")
parser.add_argument('--base_model', type=str,
                    default="paraphrase-multilingual-MiniLM-L12-v2", help='config path')
parser.add_argument('--tok_emb', action='store_true', help='', required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    sem_model = create_sign_sem_model(args.base_model, args.input_size)
    print(f"model params : {sum(p.numel() for p in sem_model.parameters() if p.requires_grad)}")
    sem_model.load_from_local(args.ckpt)
    print("loaded ..", args.ckpt)
    for pth in tqdm(glob.glob(args.in_path + "/*")):
        data = get_iter_gzip(pth)
        signs = []
        for i in data:
            signs.append(i["sign"])
        if args.tok_emb:
            signs = sem_model.sign_encode(signs, convert_to_numpy=True, normalize_embeddings=True,
                                          output_value="token_embeddings")
        else:
            signs = sem_model.sign_encode(signs, convert_to_numpy=True)  # )
        out = []
        for x, i in enumerate(data):
            i["sign"] = signs[x]
            out.append(i)
        with gzip.open(pth.replace(args.in_path, args.out_path), "wb") as f:
            pickle.dump(out, f)
        print("saved >>> ", pth)
