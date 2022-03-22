from BPE_apply import bpe_apply
from BPE_train import bpe_train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_text", type=str, default="data/smalltrain.txt")
    parser.add_argument("--save_vocab", type=str, default="data/1000.txt")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--test_text", type=str)
    parser.add_argument("--vocab_filename", type=str)
    parser.add_argument("--answer_filename", type=str, default="data/answer.txt")
    args = parser.parse_args()

    if args.mode == "train":
        bpe_train(args.train_text, args.save_vocab, args.vocab_size, args.min_frequency)
    elif args.mode == "apply":
        bpe_apply(args.test_text, args.vocab_filename, args.answer_filename)
