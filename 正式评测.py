import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from 数据准备 import SEEDS, VisualAutoencoder, collect_dataset, image_batch
from 行为克隆 import BCPolicy, evaluate, shortest_lengths
from 世界模型 import RSSM, prediction_errors
from 记忆模型 import CEMRSSM, evaluate_cem


TRAIN_SEEDS = (0, 1, 2)
SOURCES = (("A", "行为克隆", "bc"), ("B", "RSSM", "rssm"), ("C", "CEM", "cem"))
GROUPS = ("A", "B", "C", "D")
METRICS = ("sr", "grid_spl", "action_efficiency", "wrong_goal_rate",
           "timeout_rate", "collision_rate", "mean_time_s")
EPISODE_FIELDS = ("episode_id", "outcome", "terminated", "truncated", "moves", "optimal_moves",
                  "steps", "optimal_actions", "forward_attempts", "collisions", "grid_spl",
                  "action_efficiency", "time_s", "reads", "writes")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoints(root, encoder_hash):
    checkpoints, versions = {}, []
    for group, folder, name in SOURCES:
        for seed in TRAIN_SEEDS:
            path = root / folder / f"{name}_seed_{seed}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["seed"] != seed or saved["encoder_sha256"] != encoder_hash:
                raise ValueError(f"权重的种子或编码器不匹配：{path}")
            checkpoints[group, seed] = saved
            entry = {"group": group, "seed": seed, "epoch": saved["epoch"],
                     "file": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            if group == "C":
                entry["memory_config"] = saved["memory_config"]
                entry["memory_sha256"] = hashlib.sha256(saved["state_dict"]["memory"].numpy().tobytes()).hexdigest()
                entry["used_by"] = ["C", "D"]
            versions.append(entry)
    return checkpoints, versions


@torch.no_grad()
def fixed_diagnostic_episodes(tasks, encoder, device):
    seed = SEEDS["sampling"] + 2
    raw = collect_dataset(tasks, 0, seed)
    episodes, actions = [], []
    for ep in raw:
        features = encoder(image_batch(ep["rgb"], device)).cpu()
        episodes.append({"features": features, "actions": ep["actions"]})
        actions.append({"episode_id": ep["episode_id"], "actions": ep["actions"].tolist(),
                        "outcome": ep["outcome"]})
    return episodes, {"sampling_seed": seed, "episodes": actions}


def summarize(seeds):
    rows = []
    for group in GROUPS:
        selected = [row for row in seeds if row["group"] == group]
        for key in METRICS:
            values = [row[key] for row in selected if row[key] is not None]
            rows.append({"group": group, "metric": key, "n_seeds": len(values),
                         "mean": mean(values) if values else None,
                         "sample_sd": stdev(values) if len(values) > 1 else None})
    return rows


def align_pairs(seed, results):
    rows = []
    for c, d in zip(results["C"]["episodes"], results["D"]["episodes"]):
        if c["episode_id"] != d["episode_id"]:
            raise ValueError("C 和 D 的任务次序不一致")
        rows.append({"seed": seed, "episode_id": c["episode_id"],
                     "outcome_C": c["outcome"], "outcome_D": d["outcome"],
                     "success_delta": int(c["outcome"] == "success") - int(d["outcome"] == "success"),
                     "spl_delta": c["grid_spl"] - d["grid_spl"],
                     "actions_differ": c["actions"] != d["actions"]})
    return rows


def run(root, output, split="test", limit=None):
    root, output = Path(root), Path(output)
    data = root / "数据"
    manifest_path, encoder_path = data / "manifest.json", data / "encoder.pt"
    manifest_hash, encoder_hash = sha256(manifest_path), sha256(encoder_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [dict(task, start=tuple(task["start"])) for task in manifest[split][:limit]]
    optimal = [shortest_lengths(task) for task in tasks]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
    if saved_encoder["manifest_sha256"] != manifest_hash:
        raise ValueError("编码器和任务清单不匹配")
    encoder = VisualAutoencoder().encoder.to(device).eval().requires_grad_(False)
    encoder.load_state_dict(saved_encoder["state_dict"])
    checkpoints, versions = load_checkpoints(root, encoder_hash)

    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "配置.json", {
        "split": split, "task_count": len(tasks), "task_ids": [task["episode_id"] for task in tasks],
        "seeds": TRAIN_SEEDS, "device": str(device), "torch": str(torch.__version__),
        "manifest_sha256": manifest_hash, "encoder_sha256": encoder_hash,
        "selection": "validation SR, then grid SPL, then earliest epoch",
        "C_write_enabled": True, "D_write_enabled": False,
        "checkpoints": versions, "evaluation_sha256": sha256(Path(__file__)),
        "sources": {name: sha256(root / name) for name in
                    ("环境缓存.py", "数据准备.py", "行为克隆.py", "世界模型.py", "记忆模型.py")},
    })
    diagnostic, action_log = fixed_diagnostic_episodes(tasks, encoder, device)
    save_json(output / "固定动作.json", action_log)
    episode_rows, seed_rows, pairs, prediction_rows = [], [], [], []

    for (group, seed), checkpoint in checkpoints.items():
        if group == "C":
            model = CEMRSSM(**checkpoint["memory_config"]).to(device)
        else:
            model = (BCPolicy if group == "A" else RSSM)().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        if group == "C":
            results = {name: evaluate_cem(model, encoder, tasks, optimal, device, enabled)
                       for name, enabled in (("C", True), ("D", False))}
            pairs.extend(align_pairs(seed, results))
        else:
            metrics, episodes = evaluate(model, encoder, tasks, optimal, device)
            results = {group: {"metrics": metrics, "episodes": episodes}}

        if group in ("B", "C"):
            prediction_group = "C/D" if group == "C" else "B"
            for horizon, result in prediction_errors(model, diagnostic, device).items():
                prediction_rows.append({"group": prediction_group, "seed": seed,
                                        "horizon": horizon, **result})

        for name, result in results.items():
            metrics = result["metrics"]
            prefix = {"group": name, "seed": seed, "epoch": checkpoint["epoch"]}
            save_json(output / f"{name}_seed_{seed}.json", {**prefix, **result})
            seed_rows.append({**prefix, "n_tasks": len(result["episodes"]),
                              **metrics["counts"], **{key: metrics[key] for key in METRICS}})
            for ep in result["episodes"]:
                episode_rows.append({**prefix, **{key: ep.get(key) for key in EPISODE_FIELDS}})
            print(f"{name} seed={seed} success={metrics['counts']['success']}/{len(tasks)} "
                  f"SPL={metrics['grid_spl']:.4f}", flush=True)

    summary = summarize(seed_rows)
    paired_seeds = []
    for seed in TRAIN_SEEDS:
        rows = [row for row in pairs if row["seed"] == seed]
        paired_seeds.append({"seed": seed, "n_tasks": len(rows),
                             "sr_delta": mean(row["success_delta"] for row in rows),
                             "spl_delta": mean(row["spl_delta"] for row in rows),
                             "different_actions": sum(row["actions_differ"] for row in rows)})
    save_csv(output / "逐回合.csv", episode_rows)
    save_csv(output / "逐种子.csv", seed_rows)
    save_csv(output / "汇总.csv", summary)
    save_csv(output / "C_D逐回合配对.csv", pairs)
    save_csv(output / "C_D逐种子配对.csv", paired_seeds)
    save_csv(output / "预测诊断.csv", prediction_rows)
    return summary


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run(folder, folder / "正式评测结果")