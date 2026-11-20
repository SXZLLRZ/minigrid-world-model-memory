import json
from pathlib import Path
from time import perf_counter

import torch
from torch.distributions import kl_divergence
from torch.nn import functional as F

from 世界模型 import LOSS_WEIGHTS, RSSM, batch_transitions
from 数据准备 import VisualAutoencoder
from 环境缓存 import IGNORE, T
from 行为克隆 import evaluate, shortest_lengths


class CEMRSSM(RSSM):
    def __init__(self, alpha=0.3, beta=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("memory", F.normalize(torch.randn(128, 128), dim=-1))

    def retrieve(self, context, memory):
        snapshot = memory.detach().clone()
        similarity = F.normalize(context, dim=-1) @ F.normalize(snapshot, dim=-1).T
        scores, slots = similarity.topk(4, dim=-1)
        weights = scores.softmax(dim=-1)[..., None]
        recalled = (weights * snapshot[slots]).sum(dim=1)
        return context.lerp(recalled, self.alpha), slots

    @torch.no_grad()
    def write(self, context, slots, memory):
        selected = slots[:, 0]
        before = memory.index_select(0, selected)
        updated = before.lerp(context.detach(), self.beta)
        memory.index_copy_(0, selected, updated)
        return updated - before

    def forecasts(self, state):
        batch = state[0].size(0)
        actions = torch.arange(3, device=state[0].device).repeat(batch)
        repeated = tuple(part.repeat_interleave(3, dim=0) for part in state)
        future = self.decode(self.imagine(repeated, actions)).reshape(batch, 3, 64)
        choices = F.one_hot(actions, 3).float().reshape(batch, 3, 3)
        return future, choices

    def score_row(self, raw, goal, choices, future, memory):
        context, slots = self.retrieve(raw, memory)
        context = context[:, None, :].expand(-1, 3, -1)
        goals = F.one_hot(goal, 3).float()[:, None, :].expand(-1, 3, -1)
        inputs = torch.cat((context, goals, choices, future), dim=-1)
        return self.score(inputs).squeeze(-1), slots


def cem_chunk(model, batch, lengths, start, state):
    features, goals, previous, actual, future, targets, valid = batch
    end = min(start + T, actual.size(1))
    part = slice(start, end)
    rows = torch.arange(actual.size(0), device=actual.device)
    padding_score = features.new_zeros(3)
    reconstructions, predictions, divergences, scores = [], [], [], []
    writes = 0
    for t in range(start, end):
        state, prior, posterior = model.observe(features[:, t], previous[:, t], state)
        reconstructions.append(model.decode(state))
        candidates, choices = model.forecasts(state)
        predictions.append(candidates[rows, actual[:, t]])
        divergences.append(kl_divergence(posterior, prior).mean(-1))
        context = model.context(torch.cat(state, dim=-1))
        step_scores = [padding_score] * len(lengths)
        for row, length in enumerate(lengths):
            if t < length:
                i = slice(row, row + 1)
                raw = context[i]
                logits, slots = model.score_row(raw, goals[i], choices[i], candidates[i], model.memory)
                step_scores[row] = logits[0]
                model.write(raw, slots, model.memory)
                writes += 1
        scores.append(torch.stack(step_scores))

    mask = valid[:, part]
    labels = targets[:, part]
    bc_mask = labels != IGNORE
    logits = torch.stack(scores, dim=1)
    recon_error = (torch.stack(reconstructions, dim=1) - features[:, part]).square().mean(-1)
    pred_error = (torch.stack(predictions, dim=1) - future[:, part]).square().mean(-1)
    ce_sum = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1),
                             ignore_index=IGNORE, reduction="sum")
    losses = {"recon": recon_error[mask].mean(), "pred": pred_error[mask].mean(),
              "kl": torch.stack(divergences, dim=1)[mask].mean(),
              "bc": ce_sum / bc_mask.sum().clamp_min(1)}
    correct = int(((logits.argmax(-1) == labels) & bc_mask).sum())
    return losses, state, int(mask.sum()), int(bc_mask.sum()), correct, writes


def train_epoch(model, episodes, device, generator, optimizer):
    model.train()
    order = torch.randperm(len(episodes), generator=generator).tolist()
    summed = dict.fromkeys(LOSS_WEIGHTS, 0.0)
    transitions, labels, correct, updates, writes = 0, 0, 0, 0, 0
    for offset in range(0, len(order), 8):
        group = [episodes[i] for i in order[offset:offset + 8]]
        lengths = [len(ep["actions"]) for ep in group]
        batch = batch_transitions(group, device)
        state = None
        for start in range(0, batch[0].size(1), T):
            losses, state, n, n_bc, hit, n_writes = cem_chunk(model, batch, lengths, start, state)
            state = tuple(part.detach() for part in state)
            loss = sum(LOSS_WEIGHTS[key] * value for key, value in losses.items())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in losses.items():
                summed[key] += value.item() * (n_bc if key == "bc" else n)
            transitions += n
            labels += n_bc
            correct += hit
            updates += 1
            writes += n_writes
    averages = {key: value / (max(labels, 1) if key == "bc" else transitions)
                for key, value in summed.items()}
    weighted = {key: LOSS_WEIGHTS[key] * value for key, value in averages.items()}
    return {"losses": averages, "weighted": weighted, "total": sum(weighted.values()),
            "transitions": transitions, "labels": labels, "accuracy": correct / labels if labels else None,
            "updates": updates, "writes": writes}


class CEMPolicy:
    def __init__(self, model, write_enabled):
        self.model = model
        self.write_enabled = write_enabled
        self.traces = []

    def eval(self):
        self.model.eval()

    def __call__(self, features, goal_ids, previous_actions, hidden=None):
        if hidden is None:
            state, memory, trace = None, self.model.memory.detach().clone(), []
            self.traces.append(trace)
        else:
            state, memory, trace = hidden

        state, _, _ = self.model.observe(features[:, 0], previous_actions[:, 0], state)
        raw = self.model.context(torch.cat(state, dim=-1))
        future, choices = self.model.forecasts(state)
        logits, slots = self.model.score_row(raw, goal_ids, choices, future, memory)
        action = int(logits[0].argmax())
        read_slots = slots[0].tolist()
        delta = self.model.write(raw, slots, memory) if self.write_enabled else None
        record = {"action": action, "read_slots": read_slots,
                  "write_slot": read_slots[0] if self.write_enabled else None,
                  "write_l2": delta.norm().item() if delta is not None else 0.0}
        if len(self.traces) == 1:
            record.update(context=raw[0].tolist(), observed=features[0, 0].tolist(),
                          predicted=future[0, action].tolist())
        trace.append(record)
        return logits[:, None], (state, memory, trace)


def evaluate_cem(model, encoder, tasks, optimal, device, write_enabled):
    policy = CEMPolicy(model, write_enabled)
    metrics, episodes = evaluate(policy, encoder, tasks, optimal, device)
    for episode, trace in zip(episodes, policy.traces):
        episode["cem"] = trace
        episode["reads"] = len(trace)
        episode["writes"] = sum(item["write_slot"] is not None for item in trace)
    return {"metrics": metrics, "episodes": episodes}


def run(data, output, epochs=6, seeds=(0, 1, 2)):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = torch.load(data / "train_features.pt", map_location="cpu", weights_only=True)
    tasks = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["val"]
    optimal = [shortest_lengths(task) for task in tasks]
    encoder = VisualAutoencoder().encoder.to(device).eval().requires_grad_(False)
    encoder.load_state_dict(torch.load(data / "encoder.pt", map_location="cpu", weights_only=True)["state_dict"])

    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = CEMRSSM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history, best_score = [], (-1, -1)
        for epoch in range(1, epochs + 1):
            start = perf_counter()
            training = train_epoch(model, train["episodes"], device, generator, optimizer)
            c = evaluate_cem(model, encoder, tasks, optimal, device, write_enabled=True)
            metrics = c["metrics"]
            score = (metrics["sr"], metrics["grid_spl"])
            if score > best_score:
                best_score = score
                d = evaluate_cem(model, encoder, tasks, optimal, device, write_enabled=False)
                torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch,
                            "memory_config": {"alpha": model.alpha, "beta": model.beta},
                            "loss_weights": LOSS_WEIGHTS, "encoder_sha256": train["encoder_sha256"]},
                           output / f"cem_seed_{seed}.pt")
                (output / f"val_seed_{seed}.json").write_text(
                    json.dumps({"epoch": epoch, "C": c, "D": d}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            history.append({"epoch": epoch, "train": training, "closed_loop_C": metrics,
                            "elapsed_s": perf_counter() - start})
            print(f"seed={seed} epoch={epoch} loss={training['total']:.4f} "
                  f"writes={training['writes']} val_sr_C={metrics['sr']:.4f}", flush=True)
        (output / f"history_seed_{seed}.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run(folder / "数据", folder / "CEM")