"""GEIAAttacker: generative embedding inversion attack, per Li, Xu, Song
(Findings of ACL 2023), "Sentence Embedding Leaks More Information than You
Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence"
(https://aclanthology.org/2023.findings-acl.881/, arXiv:2305.03010).

GEIA treats the sentence embedding as the initial pseudo-token representation
fed to a causal decoder LM: a single trainable linear projection maps the
embedding into the decoder's hidden size, that projected vector is prepended
as the first input embedding, and the decoder (fine-tuned end-to-end together
with the projection) is trained with the standard causal-LM next-token
objective over the remaining (true) text tokens. At attack time, the
projected test embedding seeds generation and the decoder autoregressively
produces the reconstruction.

This is a from-scratch reimplementation (not a port of the reference repo at
github.com/HKUST-KnowComp/GEIA) built to fit this codebase's
InversionAttacker interface and cached-embedding pipeline. The reference
paper uses GPT-2/DialoGPT-style decoders; we use GPT-2 (small) by default,
matching their lowest-cost configuration.
"""

from __future__ import annotations

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from attackers.base import InversionAttacker
import torch
import torch.nn as nn
import torch.nn.functional as F


class GEIAAttacker(InversionAttacker):
    def __init__(
        self,
        embedding_dim: int,
        decoder_model_name: str = "gpt2",
        max_length: int = 32,
        lr: float = 5e-5,
        batch_size: int = 16,
        epochs: int = 15,
        device: str | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name).to(self.device)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        hidden_size = self.decoder.config.n_embd

        # The sole embedding_transform: projects a (normalised) sentence
        # embedding into the decoder's token-embedding space, giving it a
        # single "pseudo-token" prefix -- GEIA's core mechanism.
        self.embedding_transform = nn.Linear(embedding_dim, hidden_size).to(self.device)

    def _prefix_embeds(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalised = F.normalize(embeddings.to(self.device), p=2, dim=1)
        return self.embedding_transform(normalised).unsqueeze(1)  # [B, 1, hidden]

    def fit(self, alignment_embeddings: torch.Tensor, alignment_texts: list[str]) -> None:
        self.decoder.train()
        self.embedding_transform.train()
        optimizer = torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.embedding_transform.parameters()),
            lr=self.lr,
        )

        n = alignment_embeddings.shape[0]
        wte = self.decoder.get_input_embeddings()

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_embeddings = alignment_embeddings[idx]
                batch_texts = [alignment_texts[i] for i in idx.tolist()]

                # Reserve one slot for an explicit EOS token before padding,
                # so the model has a genuine "stop here" signal to learn
                # (pad_token is aliased to eos_token below, so padding
                # positions must stay distinguishable from a true sentence
                # end only via being masked out of the loss, not by relying
                # on an EOS the model was never shown).
                raw = self.tokenizer(
                    batch_texts, truncation=True, max_length=self.max_length - 1
                )["input_ids"]
                with_eos = [ids + [self.tokenizer.eos_token_id] for ids in raw]
                tokens = self.tokenizer.pad(
                    {"input_ids": with_eos}, padding="max_length",
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                input_ids = tokens["input_ids"]
                attn_mask = tokens["attention_mask"]

                text_embeds = wte(input_ids)  # [B, L, hidden]
                prefix = self._prefix_embeds(batch_embeddings)  # [B, 1, hidden]
                inputs_embeds = torch.cat([prefix, text_embeds], dim=1)  # [B, L+1, hidden]
                full_attn_mask = torch.cat(
                    [torch.ones(attn_mask.shape[0], 1, device=self.device, dtype=attn_mask.dtype), attn_mask],
                    dim=1,
                )

                # Labels: no loss on the prefix position (-100), then the true
                # token ids shifted implicitly by GPT2LMHeadModel's internal
                # shift-by-one; padded positions also excluded via -100, using
                # attn_mask (not token-id equality) since the real EOS token
                # we appended above shares its id with pad_token and must not
                # be masked out.
                text_labels = input_ids.masked_fill(attn_mask == 0, -100)
                prefix_label = torch.full(
                    (input_ids.shape[0], 1), -100, dtype=torch.long, device=self.device
                )
                labels = torch.cat([prefix_label, text_labels], dim=1)

                optimizer.zero_grad()
                outputs = self.decoder(
                    inputs_embeds=inputs_embeds, attention_mask=full_attn_mask, labels=labels
                )
                outputs.loss.backward()
                optimizer.step()

                epoch_loss += outputs.loss.item()
                num_batches += 1

            print(f"  [GEIA] epoch {epoch + 1}/{self.epochs}  loss={epoch_loss / num_batches:.4f}")

    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        self.decoder.eval()
        self.embedding_transform.eval()
        reconstructions: list[str] = []
        with torch.no_grad():
            for start in range(0, test_embeddings.shape[0], self.batch_size):
                batch = test_embeddings[start : start + self.batch_size]
                prefix = self._prefix_embeds(batch)  # [B, 1, hidden]
                generated_ids = self.decoder.generate(
                    inputs_embeds=prefix,
                    attention_mask=torch.ones(prefix.shape[0], 1, device=self.device, dtype=torch.long),
                    max_new_tokens=self.max_length,
                    num_beams=3,
                    repetition_penalty=2.0,
                    length_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                reconstructions += [text.strip() for text in decoded]
        return reconstructions


if __name__ == "__main__":
    torch.manual_seed(0)

    embedding_dim = 16
    texts = [
        "the cat sat on the mat",
        "a dog ran across the yard",
        "the sun rose over the mountains",
        "she read a book by the fire",
    ]
    embeddings = torch.randn(len(texts), embedding_dim)

    attacker = GEIAAttacker(embedding_dim=embedding_dim, epochs=2, batch_size=2)
    attacker.fit(embeddings, texts)

    test_embeddings = torch.randn(2, embedding_dim)
    reconstructions = attacker.attack(test_embeddings)

    assert len(reconstructions) == test_embeddings.shape[0]
    assert all(isinstance(r, str) for r in reconstructions)
    for r in reconstructions:
        print(repr(r))
    print("smoke test passed")
