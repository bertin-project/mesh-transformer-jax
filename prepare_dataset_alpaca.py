# coding: utf-8
from prepare_dataset import *


def generate_prompt(sample):
    if sample["input"]:
        promtp = f"""A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete adecuadamente lo que se pide.

### Instrucción:
{sample["instruction"]}

### Entrada:
{sample["input"]}

### Respuesta:
{sample["output"]}"""
    else:
        promtp = f""""A continuación hay una instrucción que describe una tarea. Escribe una respuesta que complete adecuadamente lo que se pide.

### Instrucción:
{sample["instruction"]}

### Respuesta:
{sample["output"]}"""
    sample["prompt"] = promtp
    return sample

def main_alpaca(args):
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
        streaming=False,
        use_auth_token=True,
    )
    if not args.preserve_data_order:
        ds = ds.shuffle(seed=args.seed)
    ds = ds.map(generate_prompt, desc="Generating prompts")
    ds = ds.map(lambda x: tokenizer(x["prompt"]), batched=True, desc="Tokenizing", num_proc=16)
    ds.set_epoch = ds.shuffle
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


if __name__ == "__main__":
    main_alpaca(parse_args())

