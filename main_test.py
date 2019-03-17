# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
from dataset import collate_fn, SeqDataset
from transformer.Translator import Translator
from preprocess import load_file, convert_w2id_seq


def main():
    parser = argparse.ArgumentParser(description='main_test.py')

    parser.add_argument('-model', required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('-src', required=True,
                        help='Source test data to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source data to extract vocabs (one line per sequence)')
    parser.add_argument('-output_dir', required=True,
                        help="Dir to store the decoded outputs")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Prepare DataLoader
    preprocess_data = torch.load(args.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = load_file(args.src, preprocess_settings.max_word_seq_len, preprocess_settings.keep_case)
    test_src_insts = convert_w2id_seq(test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=collate_fn)

    translator = Translator(args)

    with open(args.output_dir + '/test_out.txt', 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line + '\n')
    print('[Info] Finished.')


if __name__ == "__main__":
    main()
