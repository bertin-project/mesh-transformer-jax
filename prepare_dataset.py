import argparse
import os
import re
import random

from pathlib import Path
from typing import List

import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import datasets
from itertools import islice

def iter_tokens(input_ids, eos_token_id):
    for token_ids in input_ids:
        for token_id in token_ids:
            yield (token_id)
        yield (eos_token_id)


def split_every_with_padding(n, iterable, pad_token_type_id=None):
    """Splits iterable every n and fills the last chunk with pad_token_type_id
    if neccessary"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        if len(piece) < n:
            piece += [pad_token_type_id] * (n - len(piece))
        yield piece
        piece = list(islice(i, n))


def split_every(n, iterable):
    """Splits iterable in chunks of n ignoring the last chunk if not long enough"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece and len(piece) == n:
        yield piece
        piece = list(islice(i, n))


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {"text": _int64_feature(data)}
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_tfrecord(sequences, fp):
    with tf.io.TFRecordWriter(fp) as writer:
        for seq in sequences:
            write_to_file(writer, seq)


def main():
    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20  # disables a misleading warning
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    epochs = 3
    seq_length = 2048

    ncc = datasets.load_dataset("NbAiLab/NCC", split="train", streaming=True, use_auth_token=True)
    ncc = ncc.map(lambda x: tokenizer(x["text"]), batched=True)
    total = epochs * len(ncc['input_ids'])
    seqs = tqdm(
        split_every(seq_length, iter_tokens(datasets.concatenate_datasets(epochs * ncc)["input_ids"], tokenizer.eos_token_id)),
        desc="Writing token ids as TF records",
        total=total
    )
    write_tfrecord(seqs, f"ncc_{total}.tfrecords")


if __name__ == "__main__":
    main()
