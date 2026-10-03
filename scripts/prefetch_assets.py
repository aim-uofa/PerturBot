"""Prefetch public model assets. No private mirrors, proxies, or credentials."""

import tyro

from openpi.shared import download


def main(weights: str = "gs://openpi-assets/checkpoints/pi05_base", *, tokenizer_only: bool = False):
    tokenizer = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    print(f"Tokenizer: {tokenizer}")
    if not tokenizer_only:
        checkpoint = download.maybe_download(weights, gs={"token": "anon"})
        print(f"Base weights: {checkpoint}")
        print("Fine-tuning weight-loader path:", checkpoint / "params")


if __name__ == "__main__":
    tyro.cli(main)
