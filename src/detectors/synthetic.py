"""Synthetic responses for testing detectors and for the judge audit.

The released hacking set stores the placeholder string "random_gibberish_1000chars"
in every gibberish row instead of actual gibberish. random_gibberish() produces the
real thing: random printable ASCII characters, seeded for reproducibility.
"""

import random
import string

PRINTABLE = string.ascii_letters + string.digits + string.punctuation + " "


def random_gibberish(seed, n_chars=1000):
    rng = random.Random(seed)
    return "".join(rng.choice(PRINTABLE) for _ in range(n_chars))


def random_letters(seed, n_chars=1000):
    rng = random.Random(seed)
    return "".join(rng.choice(string.ascii_lowercase + " ") for _ in range(n_chars))


def token_salad(seed, tokenizer, n_tokens):
    """Decode random token ids: the multilingual word salad a derailed LLM produces."""
    rng = random.Random(seed)
    vocab_size = len(tokenizer) - 300  # skip the special tokens at the end of the vocabulary
    return tokenizer.decode([rng.randrange(vocab_size) for _ in range(n_tokens)])
