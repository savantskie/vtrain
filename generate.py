"""
Generate text from a trained checkpoint.
"""

import numpy as np
import kp
import argparse
from vtrain.data.dataset         import CharDataset
from vtrain.model.lm             import SmallLM
from vtrain.model.checkpoint     import load, params_from_block

parser = argparse.ArgumentParser(description="Generate text from a trained VTrain checkpoint")
parser.add_argument("--vocab",       required=True,          help="Path to vocab.json")
parser.add_argument("--checkpoint",  required=True,          help="Path to checkpoint directory")
parser.add_argument("--d-model",     type=int, default=128,  help="Model dimension")
parser.add_argument("--n-heads",     type=int, default=4,    help="Attention heads")
parser.add_argument("--n-layers",    type=int, default=2,    help="Transformer layers")
parser.add_argument("--seed",        type=str, default="The history of", help="Seed text")
parser.add_argument("--n-chars",     type=int, default=500,  help="Characters to generate")
parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
parser.add_argument("--device",      type=int, default=0,    help="Vulkan device index")
args = parser.parse_args()

# Load vocab
ds = CharDataset.from_vocab_file(args.vocab)
print(f"Vocab loaded: {ds.vocab_size} characters")

# Rebuild model
model  = SmallLM(vocab_size=ds.vocab_size, d_model=args.d_model,
                 n_heads=args.n_heads, n_layers=args.n_layers)
params = params_from_block(model, prefix="model")

# Load weights
load(params, args.checkpoint)

# Generate
mgr    = kp.Manager(args.device)
output = model.generate(mgr, args.seed, ds, n_chars=args.n_chars, temperature=args.temperature)

print(f"\nSeed: {args.seed!r}")
print(f"\n--- Output ---\n{args.seed}{output}")