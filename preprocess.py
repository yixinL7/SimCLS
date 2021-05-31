import json
from compare_mt.rouge.rouge_scorer import RougeScorer
from multiprocessing import Pool
import os
import random
from itertools import combinations
from functools import partial
import re
import nltk
import numpy as np
import argparse

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def collect_diverse_beam_data(args):
    split = os.path.join(args.split)
    src_dir = os.path.join(args.src_dir)
    tgt_dir = os.path.join(args.tgt_dir)
    cands = []
    cands_untok = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.source.tokenized")) as src, open(os.path.join(src_dir, f"{split}.target.tokenized")) as tgt, open(os.path.join(src_dir, f"{split}.source")) as src_untok, open(os.path.join(src_dir, f"{split}.target")) as tgt_untok:
        with open(os.path.join(src_dir, f"{split}.out.tokenized")) as f_1, open(os.path.join(src_dir, f"{split}.out")) as f_2:
            for (x, y) in zip(f_1, f_2):
                x = x.strip().lower()
                cands.append(x)
                y = y.strip().lower()
                cands_untok.append(y)
                if len(cands) == args.cand_num:
                    src_line = src.readline()
                    src_line = src_line.strip().lower()
                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip().lower()
                    src_line_untok = src_untok.readline()
                    src_line_untok = src_line_untok.strip().lower()
                    tgt_line_untok = tgt_untok.readline()
                    tgt_line_untok = tgt_line_untok.strip().lower()
                    yield (src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok, os.path.join(tgt_dir, f"{cnt}.json"))
                    cands = []
                    cands_untok = []
                    cnt += 1


def build_diverse_beam(input):
    src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok, tgt_dir = input
    cands = [sent_detector.tokenize(x) for x in cands]
    abstract = sent_detector.tokenize(tgt_line)
    _abstract = "\n".join(abstract)
    article = sent_detector.tokenize(src_line)
    def compute_rouge(hyp):
        score = all_scorer.score(_abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    candidates = [(x, compute_rouge(x)) for x in cands]
    cands_untok = [sent_detector.tokenize(x) for x in cands_untok]
    abstract_untok = sent_detector.tokenize(tgt_line_untok)
    article_untok = sent_detector.tokenize(src_line_untok)
    candidates_untok = [(cands_untok[i], candidates[i][1]) for i in range(len(candidates))]
    output = {
        "article": article, 
        "abstract": abstract,
        "candidates": candidates,
        "article_untok": article_untok, 
        "abstract_untok": abstract_untok,
        "candidates_untok": candidates_untok,
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_diverse_beam_data(args):
    data = collect_diverse_beam_data(args)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_diverse_beam, data, chunksize=64))
    print("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing Parameter')
    parser.add_argument("--cand_num", type=int, default=16)
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--tgt_dir", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    make_diverse_beam_data(args)




    

