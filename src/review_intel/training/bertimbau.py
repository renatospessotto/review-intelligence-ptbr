"""Fine-tuning do BERTimbau multi-aspecto nos rótulos prata.

Receita do projeto de NLP (bertimbau-mental-health-ptbr): lr 2e-5, batch 16, warmup 10%,
e temperature scaling calibrado na validação.

    python -m review_intel.training.bertimbau --labels data/labels/silver.jsonl
"""
import argparse
import json

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from review_intel.core.data import load_labeled, split_train_val
from review_intel.core.device import best_device
from review_intel.core.io import OUTPUTS
from review_intel.models.bertimbau import BASE, L2I, MultiAspectClassifier
from review_intel.core.schema import ASPECTS


def encode(tok, df):
    enc = tok(list(df["text"]), truncation=True, max_length=256, padding=True, return_tensors="pt")
    y = torch.tensor([[L2I[r[a]] for a in ASPECTS] for _, r in df.iterrows()])
    return enc["input_ids"], enc["attention_mask"], y


def batches(ids, mask, y, bs, shuffle):
    order = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for i in range(0, len(y), bs):
        j = order[i : i + bs]
        yield ids[j], mask[j], y[j]


def multi_loss(logits, y):
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def collect_logits(model, ids, mask, y, device, bs=64):
    model.eval()
    out = [model(i.to(device), m.to(device)).cpu() for i, m, _ in batches(ids, mask, y, bs, False)]
    return torch.cat(out)


def fit_temperature(logits, y) -> torch.Tensor:
    """Uma temperatura por aspecto, minimizando o NLL na validação (logits sem temperatura)."""
    temps = []
    for k in range(len(ASPECTS)):
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure(k=k):
            opt.zero_grad()
            loss = cross_entropy(logits[:, k] / log_t.exp(), y[:, k])
            loss.backward()
            return loss

        opt.step(closure)
        temps.append(float(log_t.detach().exp().clamp(0.05, 20)))
    return torch.tensor(temps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/silver.jsonl")
    ap.add_argument("--split", default="silver_pool")
    ap.add_argument("--output", default=str(OUTPUTS / "bertimbau-multiaspecto"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    torch.manual_seed(42)
    device = best_device()

    df = load_labeled(args.split, args.labels)
    if args.limit:
        df = df.head(args.limit)
    train, val = split_train_val(df)
    tok = AutoTokenizer.from_pretrained(BASE)
    tr, va = encode(tok, train), encode(tok, val)

    model = MultiAspectClassifier(BASE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for ids, mask, y in batches(*tr, args.batch_size, True):
            loss = multi_loss(model(ids.to(device), mask.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            total += loss.item() * len(y)
        val_logits = collect_logits(model, *va, device)
        acc = (val_logits.argmax(-1) == va[2]).float().mean().item()
        print(f"época {epoch + 1}: loss treino {total / len(train):.4f} | acurácia val (por aspecto) {acc:.3f}")

    model.temperature.copy_(fit_temperature(val_logits, va[2]).to(device))
    model.save(args.output)
    summary = {"n_train": len(train), "n_val": len(val), "val_acc": acc,
               "temperatures": dict(zip(ASPECTS, model.temperature.tolist()))}
    with open(f"{args.output}/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
