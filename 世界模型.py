import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from 数据准备 import VisualAutoencoder
from 环境缓存 import BOS, IGNORE, T
from 行为克隆 import batch_episodes, evaluate, shortest_lengths


LOSS_WEIGHTS = {"recon": 1.0, "pred": 1.0, "kl": 0.05, "bc": 1.0}


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(32 + 4, 128)
        self.prior = nn.Linear(128, 2 * 32)
        self.posterior = nn.Linear(128 + 64, 2 * 32)
        self.decoder = nn.Sequential(nn.Linear(128 + 32, 128), nn.ReLU(), nn.Linear(128, 64))
        self.context = nn.Sequential(nn.Linear(128 + 32, 128), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(128 + 3 + 3 + 64, 128), nn.ReLU(), nn.Linear(128, 1))

    @staticmethod
    def gaussian(parameters):
        mean, scale = parameters.chunk(2, dim=-1)
        return Normal(mean, F.softplus(scale) + 0.1)

    def advance(self, state, actions):
        h, z = state
        action = F.one_hot(actions, 4).to(h.dtype)
        return self.rnn(torch.cat((z, action), dim=-1), h)

    def observe(self, features, previous_actions, state=None):
        if state is None:
            batch = features.size(0)
            state = (features.new_zeros(batch, 128), features.new_zeros(batch, 32))
        h = self.advance(state, previous_actions)
        prior = self.gaussian(self.prior(h))
        posterior = self.gaussian(self.posterior(torch.cat((h, features), dim=-1)))
        z = posterior.rsample() if self.training else posterior.mean
        return (h, z), prior, posterior

    def imagine(self, state, actions):
        h = self.advance(state, actions)
        z = self.gaussian(self.prior(h)).mean
        return h, z

    def decode(self, state):
        return self.decoder(torch.cat(state, dim=-1))

    def candidates(self, state, goal_ids):
        batch = goal_ids.size(0)
        actions = torch.arange(3, device=goal_ids.device).repeat(batch)
        repeated = tuple(part.repeat_interleave(3, dim=0) for part in state)
        future = self.decode(self.imagine(repeated, actions)).reshape(batch, 3, 64)
        context = self.context(torch.cat(state, dim=-1))[:, None, :].expand(-1, 3, -1)
        goals = F.one_hot(goal_ids, 3).float()[:, None, :].expand(-1, 3, -1)
        choices = F.one_hot(actions, 3).float().reshape(batch, 3, 3)
        inputs = torch.cat((context, goals, choices, future), dim=-1)
        return self.score(inputs).squeeze(-1), future

    def forward(self, features, goal_ids, previous_actions, hidden=None):
        scores = []
        for current, previous in zip(features.unbind(1), previous_actions.unbind(1)):
            hidden, _, _ = self.observe(current, previous, hidden)
            logits, _ = self.candidates(hidden, goal_ids)
            scores.append(logits)
        return torch.stack(scores, dim=1), hidden


def batch_transitions(episodes, device):
    features, goals, previous, targets = batch_episodes(episodes, device)
    actual = pad_sequence([ep["actions"] for ep in episodes], batch_first=True).to(device)
    future = pad_sequence([ep["features"][1:] for ep in episodes], batch_first=True).to(device)
    lengths = torch.tensor([len(ep["actions"]) for ep in episodes], device=device)
    valid = torch.arange(actual.size(1), device=device)[None] < lengths[:, None]
    return features, goals, previous, actual, future, targets, valid


def chunk_losses(model, batch, start, state):
    features, goals, previous, actual, future, targets, valid = batch
    end = min(start + T, actual.size(1))
    part = slice(start, end)
    rows = torch.arange(actual.size(0), device=actual.device)
    reconstructions, predictions, divergences, scores = [], [], [], []
    for t in range(start, end):
        state, prior, posterior = model.observe(features[:, t], previous[:, t], state)
        reconstructions.append(model.decode(state))
        logits, candidates = model.candidates(state, goals)
        predictions.append(candidates[rows, actual[:, t]])
        divergences.append(kl_divergence(posterior, prior).mean(-1))
        scores.append(logits)

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
    return losses, state, int(mask.sum()), int(bc_mask.sum()), correct


def rssm_pass(model, episodes, device, batch_size, generator, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(episodes), generator=generator).tolist() if training else range(len(episodes))
    summed = dict.fromkeys(LOSS_WEIGHTS, 0.0)
    transitions, labels, correct, updates = 0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for offset in range(0, len(order), batch_size):
            group = [episodes[i] for i in order[offset:offset + batch_size]]
            batch = batch_transitions(group, device)
            state = None
            for start in range(0, batch[0].size(1), T):
                losses, state, n, n_bc, hit = chunk_losses(model, batch, start, state)
                state = tuple(part.detach() for part in state)
                if training:
                    total_loss = sum(LOSS_WEIGHTS[key] * loss for key, loss in losses.items())
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    optimizer.step()
                    updates += 1
                for key, loss in losses.items():
                    summed[key] += loss.item() * (n_bc if key == "bc" else n)
                transitions += n
                labels += n_bc
                correct += hit
    averages = {key: value / (max(labels, 1) if key == "bc" else transitions)
                for key, value in summed.items()}
    weighted = {key: LOSS_WEIGHTS[key] * value for key, value in averages.items()}
    return {"losses": averages, "weighted": weighted, "total": sum(weighted.values()),
            "transitions": transitions, "labels": labels,
            "accuracy": correct / labels if labels else None, "updates": updates}


@torch.no_grad()
def prediction_errors(model, episodes, device):
    model.eval()
    sums = {1: 0.0, 5: 0.0, 10: 0.0}
    counts = dict.fromkeys(sums, 0)
    for ep in episodes:
        features = ep["features"].to(device)
        actions = ep["actions"].to(device)
        previous = torch.cat((actions.new_tensor([BOS]), actions[:-1]))
        history, state = [], None
        for t in range(len(actions)):
            state, _, _ = model.observe(features[t:t + 1], previous[t:t + 1], state)
            history.append(state)
        imagined = tuple(torch.cat(parts, dim=0) for parts in zip(*history))
        for horizon in range(1, min(10, len(actions)) + 1):
            remaining = len(actions) - horizon + 1
            imagined = model.imagine(tuple(part[:remaining] for part in imagined), actions[horizon - 1:])
            if horizon in sums:
                errors = (model.decode(imagined) - features[horizon:]).square().mean(-1)
                sums[horizon] += errors.sum().item()
                counts[horizon] += remaining
    return {h: {"mse": sums[h] / counts[h] if counts[h] else None, "count": counts[h]}
            for h in sums}


def run(data, output, epochs=6, seeds=(0, 1, 2)):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = torch.load(data / "train_features.pt", map_location="cpu", weights_only=True)
    val = torch.load(data / "val_features.pt", map_location="cpu", weights_only=True)
    task_list = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["val"]
    optimal = [shortest_lengths(task) for task in task_list]
    encoder = VisualAutoencoder().encoder.to(device).eval().requires_grad_(False)
    checkpoint = torch.load(data / "encoder.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["state_dict"])

    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = RSSM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history, best_score = [], (-1, -1)
        for epoch in range(1, epochs + 1):
            start = perf_counter()
            training = rssm_pass(model, train["episodes"], device, 8, generator, optimizer)
            validation = rssm_pass(model, val["episodes"], device, 8, generator)
            error = prediction_errors(model, val["episodes"], device)
            closed_loop, episodes = evaluate(model, encoder, task_list, optimal, device)
            record = {"epoch": epoch, "train": training, "val": validation, "prediction": error,
                      "closed_loop": closed_loop, "elapsed_s": perf_counter() - start}
            history.append(record)
            score = (closed_loop["sr"], closed_loop["grid_spl"])
            if score > best_score:
                best_score = score
                torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch,
                            "loss_weights": LOSS_WEIGHTS, "encoder_sha256": train["encoder_sha256"]},
                           output / f"rssm_seed_{seed}.pt")
                (output / f"val_seed_{seed}.json").write_text(
                    json.dumps({"epoch": epoch, "metrics": closed_loop, "prediction": error,
                                "episodes": episodes}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"seed={seed} epoch={epoch} train_loss={training['total']:.4f} "
                  f"val_pred={error[1]['mse']:.6f} val_sr={closed_loop['sr']:.4f}", flush=True)
        (output / f"history_seed_{seed}.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run(folder / "数据", folder / "RSSM")