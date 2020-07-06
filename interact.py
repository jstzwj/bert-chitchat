
import torch
import os
import json
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from itertools import islice, takewhile, repeat

from transformers import EncoderDecoderModel, BertTokenizerFast, BertModel
from dataset import TextDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_path', default='model/epoch_0/model.pth', type=str, required=False, help='模型位置')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    device = args.device
    model_path = args.model_path

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    # 打印参数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    while True:
        question = input('请输入问题：')
        ids = tokenizer.encode(question)
        input_ids = torch.tensor([ids], dtype=torch.long)
        generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
        answer = tokenizer.decode(generated[0,:])
        print(answer)
if __name__ == "__main__":
    main()