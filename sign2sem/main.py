import argparse

from sign2sem.tools.common import load_config
from sign2sem.tools.utils import train

parser = argparse.ArgumentParser(description='Train a Siamese SignBertModel')
parser.add_argument('--config', default="configs/sample_config.yaml", help='config path')
parser.add_argument("--local_rank", type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["local_rank"] = args.local_rank
    train(cfg)


