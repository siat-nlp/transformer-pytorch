# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.utils.data
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import SeqDataset, paired_collate_fn
from trainer import train


def main():
    parser = argparse.ArgumentParser(description='main_train.py')

    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-log', required=True)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    # ========= Loading Dataset ========= #
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, args)

    args.src_vocab_size = training_data.dataset.src_vocab_size
    args.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # ========= Preparing Model ========= #
    if args.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(args)

    device = torch.device('cuda' if args.cuda else 'cpu')
    transformer = Transformer(
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.max_token_seq_len,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_tgt_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout).to(device)

    args_optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(transformer, training_data, validation_data, args_optimizer, device, args)


def prepare_dataloaders(data, args):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
