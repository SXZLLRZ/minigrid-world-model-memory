import json
from collections import deque
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from 数据准备 import DIRECTIONS, INTERIOR, VisualAutoencoder, image_batch
from 环境缓存 import BOS, FORWARD, IGNORE, T, make_env


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(64 + 3 + 4, 128, batch_first=True)
        self.head = nn.Linear(128, 3)

    def forward(self, features, goal_ids, previous_actions, hidden=None):
        goals = F.one_hot(goal_ids, 3)[:, None, :].expand(-1, features.size(1), -1)
        previous = F.one_hot(previous_actions, 4)
        inputs = torch.cat((features, goals.float(), previous.float()), dim=-1)
        sequence, hidden = self.rnn(inputs, hidden)
        return self.head(sequence), hidden


def batch_episodes(episodes, device):
    features, previous, targets = [], [], []
    for episode in episodes:
        features.append(episode["features"][:-1])
        previous.append(torch.cat((torch.tensor([BOS]), episode["actions"][:-1])))
        targets.append(episode["expert_actions"].masked_fill(~episode["bc_valid"], IGNORE))
    goals = torch.tensor([episode["goal_id"] for episode in episodes], device=device)
    return (pad_sequence(features, batch_first=True).to(device), goals,
            pad_sequence(previous, batch_first=True, padding_value=BOS).to(device),
            pad_sequence(targets, batch_first=True, padding_value=IGNORE).to(device))


def bc_pass(model, episodes, device, batch_size, generator, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(episodes), generator=generator).tolist() if training else range(len(episodes))
    total_loss, correct, total, updates = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for offset in range(0, len(order), batch_size):
            group = [episodes[i] for i in order[offset:offset + batch_size]]
            features, goals, previous, targets = batch_episodes(group, device)
            hidden = None
            for start in range(0, targets.size(1), T):
                part = slice(start, start + T)
                logits, hidden = model(features[:, part], goals, previous[:, part], hidden)
                hidden = hidden.detach()
                labels = targets[:, part]
                valid = labels != IGNORE
                count = int(valid.sum())
                if count == 0:
                    continue
                loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=IGNORE)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    updates += 1
                total_loss += loss.item() * count
                correct += int(((logits.argmax(-1) == labels) & valid).sum())
                total += count
    return {"ce": total_loss / total, "accuracy": correct / total,
            "labels": total, "updates": updates}


def shortest_distance(start, goal, neighbors):
    reached, queue = {start}, deque([(start, 0)])
    while queue:
        state, distance = queue.popleft()
        if state[:2] == goal:
            return distance
        for neighbor in neighbors(state):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise ValueError("任务没有可达目标")


def shortest_lengths(task):
    goal = tuple(task["tiles"][task["goal_id"]])
    obstacles = {tuple(cell) for cell in task["walls"]}
    obstacles.update(tuple(cell) for i, cell in enumerate(task["tiles"]) if i != task["goal_id"])
    free = INTERIOR - obstacles

    def grid_neighbors(cell):
        x, y = cell
        for dx, dy in DIRECTIONS:
            if (x + dx, y + dy) in free:
                yield (x + dx, y + dy)

    def pose_neighbors(pose):
        x, y, direction = pose
        yield (x, y, (direction - 1) % 4)
        yield (x, y, (direction + 1) % 4)
        dx, dy = DIRECTIONS[direction]
        if (x + dx, y + dy) in free:
            yield (x + dx, y + dy, direction)

    start = tuple(task["start"])
    return (shortest_distance(start, goal, grid_neighbors),
            shortest_distance((*start, task["direction"]), goal, pose_neighbors))


def rollout(task, choose_action, optimal):
    task = dict(task, start=tuple(task["start"]))
    env = make_env(task["goal_id"], task)
    start_time = perf_counter()
    try:
        obs, _ = env.reset(seed=task["task_seed"])
        positions = [tuple(map(int, env.unwrapped.agent_pos))]
        actions, moves, forward_attempts, collisions = [], 0, 0, 0
        while True:
            action = choose_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            position = tuple(map(int, env.unwrapped.agent_pos))
            moves += int(position != positions[-1])
            positions.append(position)
            actions.append(action)
            forward_attempts += int(action == FORWARD)
            collisions += int(info["blocked"])
            if terminated or truncated:
                break
    finally:
        env.close()
    success = int(info["outcome"] == "success")
    best_moves, best_actions = optimal
    return {"episode_id": task["episode_id"], "outcome": info["outcome"],
            "terminated": terminated, "truncated": truncated,
            "moves": moves, "optimal_moves": best_moves, "steps": len(actions),
            "optimal_actions": best_actions, "forward_attempts": forward_attempts,
            "collisions": collisions, "grid_spl": success * best_moves / max(best_moves, moves),
            "action_efficiency": success * best_actions / max(best_actions, len(actions)),
            "time_s": perf_counter() - start_time, "actions": actions, "positions": positions}


def make_policy(model, encoder, device):
    hidden = None
    previous_action = BOS

    def choose_action(obs):
        nonlocal hidden, previous_action
        rgb = torch.from_numpy(obs["rgb"])[None]
        features = encoder(image_batch(rgb, device))[:, None]
        goal = torch.tensor([obs["goal_id"]], device=device)
        previous = torch.tensor([[previous_action]], device=device)
        logits, hidden = model(features, goal, previous, hidden)
        previous_action = int(logits[0, 0].argmax())
        return previous_action

    return choose_action


@torch.no_grad()
def evaluate(model, encoder, tasks, optimal, device):
    model.eval()
    episodes = [rollout(task, make_policy(model, encoder, device), best)
                for task, best in zip(tasks, optimal)]
    n = len(episodes)
    counts = {key: sum(ep["outcome"] == key for ep in episodes)
              for key in ("success", "wrong_goal", "timeout")}
    forward_attempts = sum(ep["forward_attempts"] for ep in episodes)
    collisions = sum(ep["collisions"] for ep in episodes)
    metrics = {"counts": counts, "sr": counts["success"] / n,
               "wrong_goal_rate": counts["wrong_goal"] / n, "timeout_rate": counts["timeout"] / n,
               "grid_spl": sum(ep["grid_spl"] for ep in episodes) / n,
               "action_efficiency": sum(ep["action_efficiency"] for ep in episodes) / n,
               "collision_rate": collisions / forward_attempts if forward_attempts else None,
               "mean_time_s": sum(ep["time_s"] for ep in episodes) / n}
    return metrics, episodes


def run(data, output, epochs=6, seeds=(0, 1, 2)):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_list = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["val"]
    optimal = [shortest_lengths(task) for task in task_list]
    train = torch.load(data / "train_features.pt", map_location="cpu", weights_only=True)
    val = torch.load(data / "val_features.pt", map_location="cpu", weights_only=True)
    encoder = VisualAutoencoder().encoder.to(device).eval().requires_grad_(False)
    checkpoint = torch.load(data / "encoder.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["state_dict"])

    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = BCPolicy().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history, best_score = [], (-1, -1)
        for epoch in range(1, epochs + 1):
            training = bc_pass(model, train["episodes"], device, 8, generator, optimizer)
            validation = bc_pass(model, val["episodes"], device, 8, generator)
            closed_loop, episodes = evaluate(model, encoder, task_list, optimal, device)
            record = {"epoch": epoch, "train": training, "val": validation,
                      "closed_loop": closed_loop}
            history.append(record)
            score = (closed_loop["sr"], closed_loop["grid_spl"])
            if score > best_score:
                best_score = score
                torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch,
                            "encoder_sha256": train["encoder_sha256"]}, output / f"bc_seed_{seed}.pt")
                (output / f"val_seed_{seed}.json").write_text(
                    json.dumps({"epoch": epoch, "metrics": closed_loop, "episodes": episodes},
                               ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"seed={seed} epoch={epoch} train_ce={training['ce']:.4f} "
                  f"val_acc={validation['accuracy']:.4f} val_sr={closed_loop['sr']:.4f}", flush=True)
        (output / f"history_seed_{seed}.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run(folder / "数据", folder / "行为克隆")