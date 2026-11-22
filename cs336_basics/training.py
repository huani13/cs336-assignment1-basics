import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.crossentropy import cross_entropy
from cs336_basics.dataloader import load_data
from cs336_basics.gradclipping import gradient_clipping
from cs336_basics.lrscheduler import lr_scheduler
from cs336_basics.transformerlm import TransformerLanguageModel

def parse_args():
    p = argparse.ArgumentParser(description="Train")

    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--data-dtype", default="uint16")

    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--cosine-steps", type=int, default=None)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--total-tokens", type=int, default=327_680_000)

    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--run-name", default=None)

    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--seed", type=int, default=13)

    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device="cpu"
    if args.max_steps is None:
        args.max_steps = args.total_tokens // (args.batch_size * args.context_length)
    if args.min_lr is None:
        args.min_lr = args.lr / 10
    if args.warmup_steps is None:
        args.warmup_steps = max(1, int(0.05 * args.max_steps))
    if args.cosine_steps is None:
        args.cosine_steps = args.max_steps
    return args


def open_tokens(path, dtype, vocab_size):
    if str(path).endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.memmap(path, dtype=np.dtype(dtype), mode="r")

    sample = np.asarray(arr[: 1_000_000])
    if sample.size and int(sample.max()) >= vocab_size:
        raise ValueError("Wrong dtype or wrong tokenizer")
    return arr

@torch.inference_mode()
def evaluate(model, val_tokens, args, autocast_ctx):
    model.eval()
    total = torch.zeros((), device=args.device)
    for _ in range(args.eval_batches):
        x, y = load_data(val_tokens, args.batch_size, args.context_length, args.device)
        with autocast_ctx():
            logits = model(x)
        total += cross_entropy(logits.float(), y)
    model.train()
    return (total / args.eval_batches).item()

class Logger:
    def __init__(self, out_dir, args):
        self.f = open(Path(out_dir) / "metrics.jsonl", "a", buffering=1)
        self.wandb = None
        if args.wandb_project:
            import wandb
            self.wandb = wandb
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    def log(self, **metrics):
        self.f.write(json.dumps(metrics) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=metrics["step"])

    def close(self):
        self.f.close()
        if self.wandb:
            self.wandb.finish()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_type = torch.device(args.device).type
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    use_bf16 = not args.no_bf16 and device_type in ("cuda", "cpu")

    def autocast_ctx():
        return torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_bf16)

    train_tokens = open_tokens(args.train_data, args.data_dtype, args.vocab_size)
    val_tokens = open_tokens(args.val_data, args.data_dtype, args.vocab_size)
    print(f"train tokens: {len(train_tokens):,} val tokens: {len(val_tokens):,}")

    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.embedding.weight.numel() if hasattr(model.embedding, "weight") else 0
    print(f"params: {n_params:,} total, {n_params-n_embed:,} non-embedding")

    opt = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    start_step = 0
    resume = args.resume
    if resume == "auto":
        resume = out_dir / "latest.pt"
        resume = resume if resume.exists() else None
    if resume:
        start_step = load_checkpoint(resume, model, opt) + 1
        print(f"resumed from {resume} at step {start_step}")

    raw_model = model
    if not args.no_compile and device_type == "cuda":
        model = torch.compile(model)

    def checkpoint(step, name):
        path = out_dir / name
        tmp = path.with_suffix(".tmp")
        save_checkpoint(raw_model, opt, step, tmp)
        os.replace(tmp, path)

    logger = Logger(out_dir, args)
    tokens_per_step = args.batch_size * args.context_length

    model.train()
    t0 = time.perf_counter()
    t_last = t0
    last_log_step = start_step - 1

    best_file = out_dir / "best_val.json"
    best_val = json.loads(best_file.read_text())["val_loss"] if resume and best_file.exists() else math.inf

    for step in range(start_step, args.max_steps):
        lr = lr_scheduler(step, args.lr, args.min_lr, args.warmup_steps, args.cosine_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = load_data(train_tokens, args.batch_size, args.context_length, args.device)
        with autocast_ctx():
            logits = model(x)
        loss = cross_entropy(logits.float(), y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(raw_model.parameters(), args.grad_clip)
        opt.step()

        is_last = step == args.max_steps - 1

        if step % args.log_every == 0 or is_last:
            loss_val = loss.item()
            now = time.perf_counter()
            tok_s = (step - last_log_step) * tokens_per_step / (now - t_last)
            t_last, last_log_step = now, step
            logger.log(step=step, wallclock=now-t0, train_loss=loss_val, lr=lr, tokens=(step+1)*tokens_per_step)
            print(f"step {step:6d} | loss {loss_val:.4f} | lr {lr:.2e} | "
                 f"{tok_s:,.0f} tok/s | {now - t0:7.1f}s")
            if not math.isfinite(loss_val):
                raise RuntimeError(f"loss is {loss_val} atstep {step}, aborting")

        if (step > 0 and step % args.eval_every == 0) or is_last:
            val_loss = evaluate(model, val_tokens, args, autocast_ctx)
            now = time.perf_counter()
            logger.log(step=step, wallclock=now-t0, val_loss=val_loss, val_ppl=math.exp(val_loss))
            print(f"step {step:6d} | val loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                checkpoint(step, "best.pt")
                best_file.write_text(json.dumps({"step": step, "val_loss": val_loss}))
            t_last = time.perf_counter()

        if (step > 0 and step % args.ckpt_every == 0) or is_last:
            checkpoint(step, "latest.pt")
            t_last = time.perf_counter()

    print(f"done in {time.perf_counter() - t0:.1f}s, best val lostt {best_val:.4f}")
    logger.close()

if __name__ == "__main__":
    main()
