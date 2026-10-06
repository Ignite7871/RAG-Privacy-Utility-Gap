# HuggingFace imports MUST come before torch imports, or CUDA DLL conflicts occur on Windows (see README.md)
import argparse
from pathlib import Path

import datasets
from sentence_transformers import SentenceTransformer

import torch

SEED = 42
N_TOTAL = 50_000
N_ALIGN = 40_000
N_TEST = 10_000

CACHE_DIR = Path(__file__).parent / "cache"

ENCODERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "mpnet": ("sentence-transformers/paraphrase-mpnet-base-v2", 768),
    "gtr": ("sentence-transformers/gtr-t5-base", 768),
    "bge": ("BAAI/bge-large-en-v1.5", 1024),
}

DATASETS = ["msmarco", "nq"]


def load_passages(dataset_name: str) -> list[str]:
    if dataset_name == "msmarco":
        ds = datasets.load_dataset(
            "microsoft/ms_marco", "v2.1", split="train", streaming=True
        )
        passages: list[str] = []
        seen = set()
        for row in ds:
            for text in row["passages"]["passage_text"]:
                if text not in seen:
                    seen.add(text)
                    passages.append(text)
                if len(passages) >= N_TOTAL:
                    break
            if len(passages) >= N_TOTAL:
                break
    elif dataset_name == "nq":
        # BEIR's passage corpus for NQ (title + text chunks), the standard
        # passage-level analog to MS MARCO's passage set.
        ds = datasets.load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
        passages = []
        seen = set()
        for row in ds:
            text = f"{row['title']} {row['text']}".strip() if row["title"] else row["text"]
            if text and text not in seen:
                seen.add(text)
                passages.append(text)
            if len(passages) >= N_TOTAL:
                break
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(passages) < N_TOTAL:
        raise RuntimeError(
            f"{dataset_name}: only collected {len(passages)} unique passages, "
            f"needed {N_TOTAL}"
        )
    return passages


def split_passages(passages: list[str]) -> dict[str, list[str]]:
    rng = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(passages), generator=rng).tolist()
    align_idx = perm[:N_ALIGN]
    test_idx = perm[N_ALIGN : N_ALIGN + N_TEST]
    return {
        "align": [passages[i] for i in align_idx],
        "test": [passages[i] for i in test_idx],
    }


def encode_split(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    embeddings = model.encode(
        texts,
        batch_size=128,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    # Some encoder checkpoints (e.g. gtr-t5-base) ship mixed-precision weights
    # and return fp16 embeddings; force fp32 so downstream linear algebra
    # (e.g. attackers/algen.py's closed-form alignment) is consistent across
    # all four encoders regardless of each checkpoint's native dtype.
    embeddings = embeddings.float()
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


def cache_path(dataset_name: str, encoder_name: str, split: str) -> Path:
    return CACHE_DIR / f"{dataset_name}_{encoder_name}_{split}.pt"


def run(dataset_names: list[str], encoder_names: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        splits: dict[str, list[str]] | None = None

        for encoder_name in encoder_names:
            align_path = cache_path(dataset_name, encoder_name, "align")
            test_path = cache_path(dataset_name, encoder_name, "test")
            if align_path.exists() and test_path.exists():
                print(f"[skip] {dataset_name}/{encoder_name} already cached")
                continue

            if splits is None:
                print(f"[load] {dataset_name}: fetching {N_TOTAL} passages")
                passages = load_passages(dataset_name)
                splits = split_passages(passages)

            model_id, dim = ENCODERS[encoder_name]
            print(f"[encode] {dataset_name}/{encoder_name} (d={dim})")
            model = SentenceTransformer(model_id)

            for split_name, texts in splits.items():
                out_path = cache_path(dataset_name, encoder_name, split_name)
                if out_path.exists():
                    continue
                embeddings = encode_split(model, texts)
                torch.save({"text": texts, "embeddings": embeddings}, out_path)
                print(f"  saved {out_path} ({embeddings.shape})")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode dataset passages with sentence-transformers encoders."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Restrict to one dataset (default: run all datasets)",
    )
    parser.add_argument(
        "--encoder",
        choices=list(ENCODERS),
        default=None,
        help="Restrict to one encoder (default: run all encoders)",
    )
    args = parser.parse_args()

    dataset_names = [args.dataset] if args.dataset else DATASETS
    encoder_names = [args.encoder] if args.encoder else list(ENCODERS)

    run(dataset_names, encoder_names)
