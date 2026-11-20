import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
from copy import deepcopy


COLORS = ("red", "green", "blue")
LEFT, RIGHT, FORWARD = 0, 1, 2
BOS, IGNORE, T = 3, -100, 16

DEFAULT_TASK = {
    "walls": [(3, 2), (3, 3), (3, 4), (5, 4)],
    "tiles": [(6, 1), (6, 6), (1, 6)],
    "start": (1, 1), "direction": 0,
    "task_seed": 16, "episode_id": "debug_fixed",
}

class ThreeColorEnv(MiniGridEnv):
    def __init__(self, goal_id, task=None):
        self.goal_id = int(goal_id)
        self.task = deepcopy(DEFAULT_TASK if task is None else task)
        self.task["goal_id"] = self.goal_id

        mission_space = MissionSpace(
            mission_func=lambda: (
                f"go to the {COLORS[self.goal_id]} tile"
            )
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=8,
            max_steps=100,
            agent_view_size=7,
            see_through_walls=False,
        )

        self.action_space = spaces.Discrete(3)
        self.episode_done = True
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x, y in self.task["walls"]:
            self.grid.set(x, y, Wall())
        for color, position in zip(COLORS, self.task["tiles"]):
            self.grid.set(*position, Floor(color))
        self.agent_pos = self.task["start"]
        self.agent_dir = self.task["direction"]
        self.mission = f"go to the {COLORS[self.goal_id]} tile"

    def reset(self,seed=None,options=None):
        result = super().reset(
            seed=seed,
            options=options
        )
        self.episode_done = False
        return result

    def step(self, action):
        action = int(action)
        old_position = tuple(self.agent_pos)
        obs, _, _, _, _ = super().step(action)
        moved = tuple(self.agent_pos) != old_position
        cell = self.grid.get(*self.agent_pos)
        entered_tile = (
            action == FORWARD
            and moved
            and isinstance(cell, Floor)
        )

        outcome = None
        if entered_tile:
            if cell.color == COLORS[self.goal_id]:
                outcome = "success"
            else:
                outcome = "wrong_goal"
        elif self.step_count >= self.max_steps:
            outcome = "timeout"

        terminated = outcome in ("success","wrong_goal")
        truncated = outcome == "timeout"

        self.episode_done = terminated or truncated

        info = {
            "outcome": outcome,
            "blocked": action == FORWARD and not moved,
            "step_count": self.step_count,
        }

        reward = float(outcome == "success")

        return obs, reward, terminated, truncated, info


class PolicyObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "rgb": env.observation_space["image"],
            "goal_id": spaces.Discrete(3),
        })

    def observation(self, observation):
        return {
            "rgb": observation["image"].copy(),
            "goal_id": self.unwrapped.goal_id,
        }


def make_env(goal_id, task=None):
    env = ThreeColorEnv(goal_id, task=task)
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    return PolicyObservation(env)


def pack_blocks(episodes):
    rows = [
        (episode_id, start)
        for episode_id, ep in enumerate(episodes)
        for start in range(0,len(ep["actions"]),T)
    ]

    count = len(rows)

    batch = {
        "rgb": torch.zeros(count, T + 1, 3, 56, 56),
        "actions": torch.full((count, T),IGNORE,dtype=torch.long),
        "previous_actions": torch.full((count, T),IGNORE,dtype=torch.long),
        "valid": torch.zeros(count, T, dtype=torch.bool),
        "obs_valid": torch.zeros(count, T + 1, dtype=torch.bool),
        "terminated": torch.zeros(count, T, dtype=torch.bool),
        "truncated": torch.zeros(count, T, dtype=torch.bool),
        "goal_ids": torch.zeros(count, dtype=torch.long),
        "expert_actions": torch.full((count, T),IGNORE,dtype=torch.long),
        "bc_valid": torch.zeros(count, T, dtype=torch.bool),
        "episode_ids": torch.tensor([row[0] for row in rows]),
        "start_steps": torch.tensor([row[1] for row in rows]),
        "episode_start": torch.tensor([row[1] == 0 for row in rows]),
    }

    for b, (episode_id, start) in enumerate(rows):
        ep = episodes[episode_id]
        end = min(start + T,len(ep["actions"]))
        n = end - start

        batch["rgb"][b, :n + 1] = (ep["rgb"][start:end + 1].permute(0, 3, 1, 2).float()/ 255)
        batch["actions"][b, :n] = (ep["actions"][start:end])
        previous = torch.cat([torch.tensor([BOS]),ep["actions"][:-1]])
        batch["previous_actions"][b, :n] = (previous[start:end])
        batch["valid"][b, :n] = True
        batch["obs_valid"][b, :n + 1] = True
        batch["terminated"][b, :n] = (ep["terminated"][start:end])
        batch["truncated"][b, :n] = (ep["truncated"][start:end])
        batch["goal_ids"][b] = ep["goal_id"]
        expert = ep.get("expert_actions", torch.full_like(ep["actions"], IGNORE))
        bc_valid = ep.get("bc_valid", torch.zeros_like(ep["actions"], dtype=torch.bool))
        batch["expert_actions"][b, :n] = expert[start:end]
        batch["bc_valid"][b, :n] = bc_valid[start:end]

    return batch

