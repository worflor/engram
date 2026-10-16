"""Train a reusable byte-coordinate Engram from an explicitly supplied local corpus.

Example: python tools/train_memory.py --input notes --format text
         --output notes.engram --corpus-id notes-v1
"""

import argparse
from itertools import islice
from pathlib import Path
import numpy as np
from engram_corpus import documents
from engram_codec import Engram
from engram_network import WellNetwork
from engram_training import LookupSource, Trainer, learn_byte_embedding


def train(args):
    if args.output.suffix != ".engram":
        raise ValueError("output must use .engram suffix")
    if (
        args.limit < 0
        or args.checkpoint_every < 1
        or args.sample_documents < 1
        or args.workers < 1
        or args.window < 1
    ):
        raise ValueError("invalid document limit, checkpoint interval or sample size")
    embedding_path = args.output.with_suffix(".embedding.npz")

    def read():
        return documents(args.input, format=args.format)

    if args.resume:
        network = WellNetwork.load(args.output)
        with np.load(embedding_path, allow_pickle=False) as archive:
            matrix = archive["matrix"]
            window = int(archive["window"])
        metadata = network.memory.metadata
        if (
            metadata.get("training_corpus_id") != args.corpus_id
            or metadata.get("training_corpus_format") != args.format
        ):
            raise ValueError("corpus identity/format differs from checkpoint")
        cursor = metadata["training_cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("invalid checkpoint cursor")
    else:
        if args.output.exists() or embedding_path.exists():
            raise ValueError(
                "output already exists; use --resume or choose a new output"
            )
        sample, chars = [], 0
        for item in islice(read(), args.sample_documents):
            sample.append(item.payload[: min(len(item.payload), 2000000 - chars)])
            chars += len(sample[-1])
            if chars >= 2000000:
                break
        matrix = learn_byte_embedding(sample, dim=args.dim)
        window = args.window
        network = WellNetwork(args.dim, novelty_radius=args.novelty_radius)
        network.memory.metadata.update(
            training_corpus_id=args.corpus_id,
            training_corpus_format=args.format,
            training_cursor=0,
        )
        cursor = 0
    source = LookupSource.bytes(matrix, window=window)
    trainer = Trainer(network, source)
    if not args.resume:
        np.savez_compressed(embedding_path, matrix=matrix, window=window)
        network.save(args.output)
    items = islice(read(), cursor, None if args.limit == 0 else cursor + args.limit)
    fitter = None
    if args.gpu:
        from engram_acceleration import GpuFitter

        fitter = GpuFitter()
    committed, skipped = 0, 0
    for event in trainer.fit(items, workers=args.workers, batch_size=32, fitter=fitter):
        committed += 1
        skipped += event.skipped
        network.memory.metadata["training_cursor"] = cursor + committed
        if committed % args.checkpoint_every == 0:
            network.save(args.output)
            print(
                f"Checkpoint: {cursor + committed} documents visited, {len(network.describe())} wells",
                flush=True,
            )
    network.memory.crystallize_dream(Engram(network.dim))
    network.save(args.output)
    print(
        f"Committed {committed} documents ({skipped} short/skipped); cursor={cursor + committed}; "
        f"wells={len(network.describe())}; checkpoint={args.output}",
        flush=True,
    )
    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["text", "jsonl", "wiki"], default="text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus-id",
        required=True,
        help="Stable caller-owned dataset/order version; must match on resume",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum additional documents; 0 reads to end",
    )
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--sample-documents", type=int, default=64)
    parser.add_argument("--novelty-radius", type=float, default=0.15)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    try:
        train(parser.parse_args())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Training failed: {error}\n")
