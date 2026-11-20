import hashlib
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from 环境缓存 import COLORS, LEFT, RIGHT, FORWARD, T, make_env, pack_blocks


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
INTERIOR = {(x, y) for x in range(1, 7) for y in range(1, 7)}
FULL_COUNTS = (128, 24, 48)
SEEDS = {"map": 1500, "split": 1501, "sampling": 1502, "encoder": 1503}


def geometry_hash(walls):
    variants = []
    for mirror in (False, True):
        for turns in range(4):
            cells = [(7 - x if mirror else x, y) for x, y in walls]
            for _ in range(turns):
                cells = [(7 - y, x) for x, y in cells]
            variants.append(tuple(sorted(cells)))
    payload = "8x8-v1|" + json.dumps(min(variants), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connected_cells(cells, start):
    reached, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            point = (x + dx, y + dy)
            if point in cells and point not in reached:
                reached.add(point)
                queue.append(point)
    return reached


def build_manifest(counts=FULL_COUNTS):
    rng, geometries = random.Random(SEEDS["map"]), {}

    while len(geometries) < sum(FULL_COUNTS):
        walls = sorted(rng.sample(sorted(INTERIOR), 4))
        free = INTERIOR - set(walls)
        if len(connected_cells(free, min(free))) == len(free):
            geometries.setdefault(geometry_hash(walls), walls)

    keys = sorted(geometries)
    random.Random(SEEDS["split"]).shuffle(keys)

    manifest, offset = {}, 0
    for split, full_count, count in zip(("train", "val", "test"), FULL_COUNTS, counts):
        group_keys = keys[offset:offset + full_count][:count]
        offset += full_count
        manifest[split] = []

        for key in group_keys:
            task_rng = random.Random(int(key[:16], 16) ^ SEEDS["map"])
            free = INTERIOR - set(geometries[key])

            while True:
                tiles = task_rng.sample(sorted(free), 3)
                ordinary = free - set(tiles)
                joined = len(connected_cells(ordinary, min(ordinary))) == len(ordinary)
                accessible = all(any((x + dx, y + dy) in ordinary
                                     for dx, dy in DIRECTIONS) for x, y in tiles)
                if joined and accessible:
                    break
            starts = task_rng.sample(sorted(ordinary), 2)
            for start_id, start in enumerate(starts):
                direction = task_rng.randrange(4)
                for goal_id in range(3):
                    manifest[split].append({
                        "geometry_hash": key, "split": split,
                        "episode_id": f"{split}_{key[:12]}_{start_id}_{goal_id}",
                        "walls": geometries[key], "tiles": tiles, "start": start,
                        "direction": direction, "goal_id": goal_id, "max_steps": 100,
                        "task_seed": int(key[:8], 16) + start_id * 3 + goal_id,
                    })
    return manifest


class PartialExpert:
    def __init__(self):
        self.known = np.full((8, 8), -1, dtype=np.int8)

    def observe(self, base):
        view, visible = base.gen_obs_grid()
        ax, ay = base.agent_pos
        dx, dy = base.dir_vec
        rx, ry = base.right_vec
        size = base.agent_view_size
        for vx in range(size):
            for vy in range(size):
                x = int(ax + rx * (vx - size // 2) + dx * (size - 1 - vy))
                y = int(ay + ry * (vx - size // 2) + dy * (size - 1 - vy))
                if not visible[vx, vy] or not (0 <= x < 8 and 0 <= y < 8):
                    continue
                cell = view.get(vx, vy)
                value = 0
                if cell is not None and cell.type == "wall":
                    value = 1
                elif cell is not None and cell.type == "floor":
                    value = 2 + COLORS.index(cell.color)
                self.known[x, y] = value

    def paths(self, pose, goal_id):
        paths, queue = {pose: ()}, deque([pose])
        while queue:
            x, y, direction = queue.popleft()
            if self.known[x, y] == 2 + goal_id:
                continue
            dx, dy = DIRECTIONS[direction]
            options = [(LEFT, (x, y, (direction - 1) % 4)),
                       (RIGHT, (x, y, (direction + 1) % 4))]
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.known[nx, ny] in (0, 2 + goal_id):
                options.append((FORWARD, (nx, ny, direction)))
            for action, state in options:
                if state not in paths:
                    paths[state] = paths[(x, y, direction)] + (action,)
                    queue.append(state)
        return paths

    def propose(self, pose, goal_id):
        paths = self.paths(pose, goal_id)
        for (x, y, _), path in paths.items():
            if path and self.known[x, y] == 2 + goal_id:
                return path[0], True, "known_goal"
        frontiers = []
        for (x, y, direction), path in paths.items():
            dx, dy = DIRECTIONS[direction]
            nx, ny = x + dx, y + dy
            if (path and self.known[x, y] == 0 and 0 <= nx < 8 and 0 <= ny < 8
                    and self.known[nx, ny] == -1):
                frontiers.append((len(path), y, x, direction, path))
        if frontiers:
            return min(frontiers)[-1][0], True, "frontier"
        return LEFT, False, "fallback"


def collect_episode(task, rng, repeat):
    task = dict(task, episode_id=f"{task['episode_id']}_r{repeat:04d}")
    env = make_env(task["goal_id"], task)
    obs, _ = env.reset(seed=task["task_seed"])
    expert = PartialExpert()
    frames, actual, labels, label_valid = [obs["rgb"]], [], [], []
    terminateds, truncateds, blocked = [], [], []
    maps, poses, sources, random_branches = [], [], [], []
    try:
        while True:
            expert.observe(env.unwrapped)
            x, y = env.unwrapped.agent_pos
            pose = (int(x), int(y), int(env.unwrapped.agent_dir))
            suggestion, valid, source = expert.propose(pose, task["goal_id"])
            random_branch = rng.random() >= 0.8
            action = rng.randrange(3) if random_branch else suggestion
            maps.append(expert.known.copy())
            poses.append(pose)
            sources.append(source)
            random_branches.append(random_branch)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(obs["rgb"])
            actual.append(action)
            labels.append(suggestion)
            label_valid.append(valid)
            terminateds.append(terminated)
            truncateds.append(truncated)
            blocked.append(info["blocked"])
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "name": task["episode_id"], "episode_id": task["episode_id"],
        "goal_id": task["goal_id"], "task": task, "outcome": info["outcome"],
        "rgb": torch.from_numpy(np.stack(frames)),
        "actions": torch.tensor(actual, dtype=torch.long),
        "expert_actions": torch.tensor(labels, dtype=torch.long),
        "bc_valid": torch.tensor(label_valid, dtype=torch.bool),
        "terminated": torch.tensor(terminateds, dtype=torch.bool),
        "truncated": torch.tensor(truncateds, dtype=torch.bool),
        "blocked": torch.tensor(blocked, dtype=torch.bool),
        "expert_maps": torch.from_numpy(np.stack(maps)),
        "expert_poses": torch.tensor(poses, dtype=torch.long),
        "expert_sources": sources,
        "random_branches": torch.tensor(random_branches, dtype=torch.bool),
    }


def collect_dataset(tasks, min_steps, seed):
    rng, episodes, steps, index = random.Random(seed), [], 0, 0
    while steps < min_steps or index < len(tasks):
        ep = collect_episode(tasks[index % len(tasks)], rng, index // len(tasks))
        episodes.append(ep)
        steps += len(ep["actions"])
        index += 1
        if index % 64 == 0:
            print(f"采集：{index}回合，{steps}个有效转移", flush=True)
    return episodes


class VisualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, rgb):
        return self.decoder(self.encoder(rgb))


def image_batch(frames, device):
    return frames.to(device).permute(0, 3, 1, 2).float() / 255


def train_encoder(episodes, device, epochs):
    torch.manual_seed(SEEDS["encoder"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEEDS["encoder"])
    frames = torch.cat([ep["rgb"] for ep in episodes])
    loader = DataLoader(TensorDataset(frames), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(SEEDS["encoder"]))
    model = VisualAutoencoder().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        loss_sum = 0.0
        for (batch,) in loader:
            rgb = image_batch(batch, device)
            loss = (model(rgb) - rgb).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(rgb)
        history.append(loss_sum / len(frames))
        print(f"编码器第{epoch + 1}轮：训练重建MSE={history[-1]:.6f}", flush=True)
    return model.eval().requires_grad_(False), history, frames


@torch.no_grad()
def cache_features(episodes, encoder, device, encoder_hash):
    cached, chunks = [], []
    for episode_id, ep in enumerate(episodes):
        features = encoder(image_batch(ep["rgb"], device)).cpu()
        packed = pack_blocks([ep])
        packed.pop("rgb")
        packed["episode_ids"].fill_(episode_id)
        packed["features"] = torch.zeros(len(packed["valid"]), T + 1, 64)
        for row, start in enumerate(packed["start_steps"].tolist()):
            n = int(packed["valid"][row].sum())
            packed["features"][row, :n + 1] = features[start:start + n + 1]
        chunks.append(packed)
        cached.append({key: value for key, value in ep.items()
                       if key not in ("rgb", "expert_maps", "expert_poses")})
        cached[-1]["features"] = features
    return {"episodes": cached,
            "blocks": {key: torch.cat([part[key] for part in chunks]) for key in chunks[0]},
            "encoder_sha256": encoder_hash, "T": T, "feature_dim": 64}


def run(output, counts=FULL_COUNTS, train_steps=25000, epochs=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(counts)
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8", newline="\n")
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    train = collect_dataset(manifest["train"], train_steps, SEEDS["sampling"])
    val = collect_dataset(manifest["val"], 0, SEEDS["sampling"] + 1)
    torch.save(train, output / "train_rgb.pt")
    torch.save(val, output / "val_rgb.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history, frames = train_encoder(train, device, epochs)
    torch.save({"state_dict": model.state_dict(), "seed": SEEDS["encoder"],
                "train_mse": history}, output / "autoencoder.pt")
    torch.save({"state_dict": model.encoder.state_dict(), "feature_dim": 64,
                "training_split": "train", "manifest_sha256": manifest_hash,
                "seed": SEEDS["encoder"]}, output / "encoder.pt")
    encoder_hash = hashlib.sha256((output / "encoder.pt").read_bytes()).hexdigest()
    del frames
    for split, episodes in (("train", train), ("val", val)):
        cache = cache_features(episodes, model.encoder, device, encoder_hash)
        torch.save(cache, output / f"{split}_features.pt")
    summary = {
        "data_version": "local_rgb_v1", "seeds": SEEDS, "geometries": counts,
        "environment_sha256": hashlib.sha256(
            Path(__file__).with_name("环境缓存.py").read_bytes()).hexdigest(),
        "data_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": manifest_hash, "encoder_sha256": encoder_hash,
        "encoder_training_split": "train", "encoder_epochs": epochs,
        "encoder_train_mse": history, "device": str(device),
        "episodes": {"train": len(train), "val": len(val)},
        "valid_transitions": {name: sum(len(ep["actions"]) for ep in data)
                              for name, data in (("train", train), ("val", val))},
        "test_tasks": len(manifest["test"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(Path(__file__).resolve().parent / "数据")