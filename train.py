
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
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='模型训练batch大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    raw_data_path = args.raw_data_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")
    
    # dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    dataset = TextDataset(tokenizer, './dataset/train.jsonl')

    # 打印参数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    # dataloader
    def pad_collate_fn(batch):
        batch_size = len(batch)
        # find longest sequence
        source_max_len = max(map(lambda x: x['source'].shape[0], batch))
        target_max_len = max(map(lambda x: x['target'].shape[0], batch))
        # pad according to max_len
        ret = {
            'source': torch.full((batch_size, source_max_len), tokenizer.pad_token_id, dtype=torch.long),
            'target': torch.full((batch_size, target_max_len), tokenizer.pad_token_id, dtype=torch.long)
        }

        for i, sample in enumerate(batch):
            sample_source = sample['source']
            sample_target = sample['target']
            ret['source'][i,:sample_source.numel()] = sample_source
            ret['target'][i,:sample_target.numel()] = sample_target
        return ret

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    
    print('start training')
    for epoch in range(epochs):
        with tqdm(total=len(dataloader), ascii=True) as t:
            for i, sample in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = sample['source']
                decoder_input_ids = sample['target']
                loss, *args = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
                # backward
                loss.backward()
                optimizer.step()

                t.set_postfix({'loss': loss.item()})
                t.update(1)
        # save model
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_epoch_dir = os.path.join(output_dir, f'epoch_{str(epoch)}')
        if not os.path.exists(output_epoch_dir):
            os.mkdir(output_epoch_dir)
        torch.save(model.state_dict(), os.path.join(output_epoch_dir, 'model.pth'))

if __name__ == "__main__":
    main()