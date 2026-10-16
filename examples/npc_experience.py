"""Learn a delayed-cue navigation task through Engram experience wells.

The same current junction witness requires different actions depending on a
cue observed earlier. No cue is retained in a special-purpose policy field.
The Minecraft lane exercises the same controller against native game movement.
"""

import argparse
import json
import numpy as np
from engram_npc import EngramNPC, WitnessSchema

SCHEMA = WitnessSchema(
    (
        "progress",
        "cue",
        "side",
        "north_clear",
        "south_clear",
        "west_clear",
        "east_clear",
        "health",
    ),
    representation="cue-corridor-v1",
)
DIRECTIONS = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}
CELLS = {(0, 0), (1, 0), (2, 0), (3, 0), (3, -1), (3, -2), (3, 1), (3, 2)}


class WitnessMaze:
    def sample(self):
        return self.observation()

    def reset(self, cue, *, reversed_goal=False):
        self.position = (0, 0)
        self.cue = cue
        self.reversed_goal = reversed_goal
        return self.observation()

    def observation(self):
        x, z = self.position
        free = {a: (x + dx, z + dz) in CELLS for a, (dx, dz) in DIRECTIONS.items()}
        return {
            "witness": [
                x / 3,
                self.cue if x == 0 else 0,
                z / 2,
                int(free["north"]),
                int(free["south"]),
                int(free["west"]),
                int(free["east"]),
                1.0,
            ],
            "actions": sorted(a for a, allowed in free.items() if allowed),
            "position": [x, z],
        }

    def step(self, action):
        dx, dz = DIRECTIONS[action]
        x, z = self.position
        if (x + dx, z + dz) in CELLS:
            self.position = (x + dx, z + dz)
        terminal = abs(self.position[1]) == 2
        expected = -self.cue if self.reversed_goal else self.cue
        reward = (
            (1.0 if np.sign(self.position[1]) == expected else -1.0)
            if terminal
            else -0.02
        )
        return self.observation(), reward, terminal


def episode(
    agent, environment, cue, *, train=True, reversed_goal=False, limit=40, planning=0
):
    observation = environment.reset(cue, reversed_goal=reversed_goal)
    agent.begin_episode()
    # Three actual time samples, with no action, prime the AR(2) history.
    for _ in range(3):
        observation = environment.sample()
        agent.observe(observation["witness"], learn=train)
    trace = []
    for step in range(limit):
        decision = agent.decide(observation["actions"], explore=train)
        next_observation, reward, terminal = environment.step(decision.action)
        truncated = step == limit - 1 and not terminal
        if truncated:
            reward = -1.0
        agent.feedback(
            decision,
            next_observation["witness"],
            reward,
            terminal=terminal or truncated,
            next_actions=next_observation["actions"],
            learn=train,
        )
        if train and planning:
            agent.dream(planning)
        trace.append(
            {
                "action": decision.action,
                "well": decision.well,
                "reward": reward,
                "position": next_observation["position"],
            }
        )
        observation = next_observation
        if terminal or truncated:
            return {
                "success": terminal and reward > 0,
                "steps": step + 1,
                "trace": trace,
            }


def study(*, seed=7, training=160, evaluation=40, dynamics_weight=1.0, planning=0):
    agent = EngramNPC(
        SCHEMA,
        seed=seed,
        dynamics_weight=dynamics_weight,
        novelty_radius=0.08,
        epsilon=0.1,
        exploration=0.25,
    )
    environment = WitnessMaze()
    learning = []
    for i in range(training):
        learning.append(
            episode(agent, environment, 1 if i % 2 else -1, planning=planning)[
                "success"
            ]
        )
    tests = [
        episode(agent, environment, 1 if i % 2 else -1, train=False)
        for i in range(evaluation)
    ]
    return agent, {
        "seed": seed,
        "training_episodes": training,
        "evaluation_episodes": evaluation,
        "success_rate": sum(t["success"] for t in tests) / evaluation,
        "mean_steps": float(np.mean([t["steps"] for t in tests])),
        "wells": len(agent.wells),
        "real_feedback_updates": agent.updates,
        "planning_updates": agent.planning_updates,
        "last_40_training_success": float(np.mean(learning[-40:])),
        "example_trace": tests[-1]["trace"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planning", type=int, default=0)
    args = parser.parse_args()
    agent, report = study(
        seed=args.seed, training=args.training, planning=args.planning
    )
    print(json.dumps(report, indent=2))
