"""
Overnight Wikipedia training run.
Designed to survive crashes — checkpoints every N steps,
resumable from last checkpoint.
"""

import sys
import signal
import numpy as np
import kp
from pathlib import Path
from vtrain.data.dataset  import CharDataset
from vtrain.model.lm      import SmallLM
from vtrain.model.checkpoint import save, params_from_block
from vtrain.loss          import cross_entropy_loss
from vtrain.optim         import Adam
from vtrain.tensor        import Tensor
import vtrain.functional as F
import time
import json

import argparse

parser = argparse.ArgumentParser(description="Train a character-level language model on a text corpus")
parser.add_argument("--data",       required=True,         help="Path to training text file")
parser.add_argument("--run-dir",    default="models/run1", help="Directory for checkpoints and logs")
parser.add_argument("--max-chars",  type=int, default=10_000_000, help="Max characters to load (default 10M)")
parser.add_argument("--seq-len",    type=int, default=64)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--d-model",    type=int, default=128)
parser.add_argument("--n-heads",    type=int, default=4)
parser.add_argument("--n-layers",   type=int, default=2)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--max-steps",  type=int, default=10_000)
parser.add_argument("--checkpoint-every", type=int, default=200)
parser.add_argument("--log-every",  type=int, default=50)
parser.add_argument("--device",     type=int, default=0, help="Vulkan device index (default 0)")
args = parser.parse_args()

CONFIG = {
    "data_path":        args.data,
    "max_chars":        args.max_chars,
    "seq_len":          args.seq_len,
    "batch_size":       args.batch_size,
    "d_model":          args.d_model,
    "n_heads":          args.n_heads,
    "n_layers":         args.n_layers,
    "lr":               args.lr,
    "max_steps":        args.max_steps,
    "checkpoint_every": args.checkpoint_every,
    "log_every":        args.log_every,
    "run_dir":          args.run_dir,
    "device":           args.device,
}

# ── Crash handler — saves checkpoint on SIGINT/SIGTERM ────────────────────────

model_ref  = None
params_ref = None
step_ref   = [0]

def emergency_save(sig, frame):
    if model_ref is not None:
        path = f"{CONFIG['run_dir']}/checkpoints/emergency_step_{step_ref[0]:06d}"
        print(f"\nSignal received — emergency save to {path}")
        save(params_ref, path)
    sys.exit(0)

signal.signal(signal.SIGINT,  emergency_save)
signal.signal(signal.SIGTERM, emergency_save)

# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(run_dir: str):
    """Return path to latest checkpoint, or None if none exist."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None, 0

    ckpts = sorted([
        d for d in ckpt_dir.iterdir()
        if d.is_dir() and 'emergency' not in d.name
    ])

    if not ckpts:
        # Check for emergency saves
        ckpts = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])

    if not ckpts:
        return None, 0

    latest = ckpts[-1]
    step   = int(latest.name.split('_')[-1])
    return str(latest), step


def main():
    global model_ref, params_ref

    run_dir  = Path(CONFIG["run_dir"])
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "loss_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("── Wiki training run ───────────────────────────────────")
    print(f"  Run dir: {run_dir}")

    # Load data
    ds = CharDataset.from_file(CONFIG["data_path"], CONFIG["max_chars"])

    # Save vocab so we can reload it later without the full text
    vocab_path = run_dir / "vocab.json"
    ds.save_vocab(str(vocab_path))

    # Build model
    model = SmallLM(
        vocab_size = ds.vocab_size,
        d_model    = CONFIG["d_model"],
        n_heads    = CONFIG["n_heads"],
        n_layers   = CONFIG["n_layers"],
    )
    params     = params_from_block(model, prefix="model")
    model_ref  = model
    params_ref = params

    print(f"  Vocab size:  {ds.vocab_size}")
    print(f"  Data size:   {len(ds.data):,} chars")
    print(f"  Parameters:  {len(model.parameters())} tensors")

    # Resume from checkpoint if one exists
    latest_ckpt, start_step = find_latest_checkpoint(CONFIG["run_dir"])
    if latest_ckpt:
        print(f"  Resuming from step {start_step}: {latest_ckpt}")
        from vtrain.model.checkpoint import load
        load(params, latest_ckpt)
    else:
        print("  Starting fresh")

    mgr = kp.Manager(CONFIG["device"])
    opt = Adam(model.parameters(), lr=CONFIG["lr"])

    step       = start_step
    step_ref[0] = step
    t_start    = time.time()
    losses     = []

    print(f"\n  Starting from step {step}, target {CONFIG['max_steps']}")
    print("─" * 55)

    while step < CONFIG["max_steps"]:
        opt.zero_grad()

        X, Y   = ds.get_batch(CONFIG["batch_size"], CONFIG["seq_len"])
        logits = model.forward(mgr, X)
        probs  = F.softmax(mgr, logits)

        Y_flat = Y.flatten().astype(np.int32)
        Y_oh   = Tensor(
            np.eye(ds.vocab_size, dtype=np.float32)[Y_flat],
            requires_grad=False
        )

        loss = cross_entropy_loss(probs, Y_oh)
        loss.backward()
        opt.step()

        loss_val    = float(loss.data)
        step       += 1
        step_ref[0] = step
        losses.append(loss_val)

        if step % CONFIG["log_every"] == 0:
            avg_loss = np.mean(losses[-CONFIG["log_every"]:])
            elapsed  = time.time() - t_start
            rate     = step / elapsed
            eta      = (CONFIG["max_steps"] - step) / rate / 3600

            print(f"  step {step:6d}  "
                  f"loss={avg_loss:.4f}  "
                  f"rate={rate:.1f} steps/s  "
                  f"ETA={eta:.1f}h")

            # Append to loss log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "loss": avg_loss,
                    "elapsed": elapsed
                }) + "\n")

        if step % CONFIG["checkpoint_every"] == 0:
            ckpt_path = str(ckpt_dir / f"step_{step:06d}")
            save(params, ckpt_path)

    # Final save
    final_path = str(ckpt_dir / f"step_{step:06d}_final")
    save(params, final_path)
    print(f"\nDone. Final checkpoint at {final_path}")


if __name__ == "__main__":
    main()
