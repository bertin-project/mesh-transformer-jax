#!/usr/bin/env python
import argparse
import re
import os
from functools import partial
from pathlib import Path
from typing import List

# import ftfy
import tensorflow as tf
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import datasets
from itertools import islice

newlines_re = re.compile(r"\n\n+")
copy_re = re.compile(r"^.*(©|\([ \t]*c[ \t]*\)|copyright|ISBN).*$", re.IGNORECASE)
notes_re = re.compile(r"[\[\(][0-9]{1,5}[\]\)]", re.IGNORECASE)


def clean_text(text):
    if isinstance(text, dict):
        return newlines_re.sub("\n\n", notes_re.sub("", copy_re.sub("", text))).strip()
    else:
        texts = text
        return [newlines_re.sub("\n\n", notes_re.sub("", copy_re.sub("", text))).strip() for text in texts]


def remove_empty_texts(sample, column):
    text = sample[column]
    if isinstance(text, str):
        return text and not text.isspace()
    else:
        texts = text
        return [text and not text.isspace() for text in texts]


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
    Writes data to tfrecord file
    """
    feature = {"text": _int64_feature(data)}
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_tfrecord(sequences, fp):
    with tf.io.TFRecordWriter(fp) as writer:
        for idx, seq in enumerate(sequences):
            write_to_file(writer, seq)
    return idx


def generate_sample(dataset, epochs, key, preserve_data_order=False):
    for epoch in range(epochs):
        if not preserve_data_order:
            dataset.set_epoch(epoch)
        for sample in dataset:
            yield sample[key]


def main(args):
    GPT2TokenizerFast.max_model_input_sizes[
        "gpt2"
    ] = 1e20  # disables a misleading warning
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    epochs = args.n_repack_epochs
    seq_length = args.sequence_length

    ds = datasets.load_dataset(
        args.dataset,
        name=args.dataset_config or None,
        split=args.dataset_split,
        streaming=args.streaming,
        use_auth_token=True,
    )
    if not args.preserve_data_order:
        print("Shuffling data")
        ds = ds.shuffle(args.dataset_buffer_size, seed=args.seed)
    if args.streaming:
        map_kwargs = {"batched": True}
    else:
        map_kwargs = {"batched": True, "num_proc": os.cpu_count()}
    if args.normalize_numbers_spaces:
        print("Normalizing spaces, notes, and page numbers")
        ds = ds.map(lambda x: {args.dataset_text_column: clean_text(x[args.dataset_text_column])}, **map_kwargs)
    if args.remove_empty:
        print("Removing empty documents")
        ds = ds.filter(partial(remove_empty_texts, column=args.dataset_text_column), **map_kwargs)
    print(ds)
    ds = ds.map(lambda x: tokenizer(x[args.dataset_text_column]), **map_kwargs)
    seqs = tqdm(
        split_every(
            seq_length,
            iter_tokens(
                generate_sample(ds, epochs, "input_ids", args.preserve_data_order), tokenizer.eos_token_id
            ),
        ),
        desc="Writing token ids as TF records",
    )
    filepath = args.output_dir / f"{args.name}.tfrecords"
    seq_count = write_tfrecord(seqs, filepath.as_posix())
    filepath_seq = args.output_dir / f"{args.name}_{seq_count}.tfrecords"
    os.rename(filepath.as_posix(), filepath_seq.as_posix())


def parse_args():
    parser = argparse.ArgumentParser(description="""
    Converts a text dataset from Huggingface into the training data format expected by the model.
    This script creates a single .tfrecords file as output
        - Why: the model's data loader ignores "trailing" data (< 1 batch) at the end of a .tfrecords file
            - this causes data loss if you have many .tfrecords files
        - This is probably not appropriate for very large datasets
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("name", type=str,
                        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count")

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset path or hub name.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str, default="",
        help="Dataset config.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str, default="train",
        help="Dataset split. It accepts any Huggingface datasets expression for splits.",
    )
    parser.add_argument(
        "--dataset_text_column",
        type=str, default="text",
        help="Dataset text field name.",
    )
    parser.add_argument(
        "--dataset_buffer_size",
        type=int, default=10_000,
        help="Dataset buffer size for shuffling.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int, default=2048,
        help="Sequence length of each TF record.",
    )
    parser.add_argument("--streaming", action="store_true", help="Whether to stream the dataset or not")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (default: current directory)")

    cleaning_args = parser.add_argument_group('data cleaning arguments')
    cleaning_args.add_argument("--remove_empty", action="store_true", help="Remove empty documents with only space characters in it")
    cleaning_args.add_argument("--normalize_numbers_spaces", action="store_true", help="Converts double breaklines to simple and remove page numbers and notes")

    # cleaning_args.add_argument("--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy")
    # cleaning_args.add_argument("--normalize-with-wikitext-detokenize",
    #                            action="store_true", help="Use wikitext detokenizer")
    # minu_help = "Exclude repetitive documents made up of < MIN_UNIQUE_TOKENS unique tokens. These can produce large gradients."
    # minu_help += " Set <= 0 to disable. If enabled, 200 is a good default value. (Default: 0)"
    # cleaning_args.add_argument("--min-unique-tokens", type=int, default=0,
    #                            help=minu_help)

    shuffle_pack_args = parser.add_argument_group('data shuffling/packing arguments')
    repack_ep_help = "Repeat the data N_REPACK_EPOCHS times, shuffled differently in each repetition. Recommended for multi-epoch training (set this to your intended number of epochs)."
    shuffle_pack_args.add_argument("--n-repack-epochs",
                                   type=int, default=1,
                                   help=repack_ep_help
                                   )
    shuffle_pack_args.add_argument("--seed", type=int, default=10,
                                   help="random seed for shuffling data (default: 10)")
    shuffle_pack_args.add_argument("--preserve-data-order",
                                   default=False, action="store_true",
                                   help="Disables shuffling, so the input and output data have the same order.")

    args = parser.parse_args()

    # convert output_dir to pathy
    args.output_dir = Path(args.output_dir)

    return args

if __name__ == "__main__":
    main(parse_args())
