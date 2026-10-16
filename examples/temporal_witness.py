"""A controlled memory-aliasing task: did a pulse occur in the last 3 samples?

Both histories have identical current witnesses and cumulative lag moments.
No last-pulse token is stored by the controller. Recency enters through the
same generic witness fits at explicitly configured temporal scales.
"""

import json
from engram_npc import EngramNPC, WitnessSchema

SCHEMA = WitnessSchema(("pulse", "companion"), representation="binary-pulse/v1")
HISTORIES = ((0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0))
ACTIONS = ("recent", "old")


def encounter(agent, values, *, train):
    agent.begin_episode()
    for i, value in enumerate(values):
        agent.observe([value, 0], learn=train and i == len(values) - 1)
    answer = "recent" if any(values[-3:]) else "old"
    decision = agent.decide(ACTIONS, explore=train)
    agent.feedback(
        decision,
        [values[-1], 0],
        1.0 if decision.action == answer else -1.0,
        terminal=True,
        learn=train,
    )
    return decision.action == answer


def study(decays, *, seed=7, training=240, evaluation=100):
    agent = EngramNPC(
        SCHEMA, history_decays=decays, novelty_radius=0.01, seed=seed, epsilon=0.1
    )
    for i in range(training):
        encounter(agent, HISTORIES[i % 2], train=True)
    updates = agent.updates
    successes = sum(
        encounter(agent, HISTORIES[i % 2], train=False) for i in range(evaluation)
    )
    assert agent.updates == updates
    return agent, {
        "decays": decays,
        "seed": seed,
        "training_encounters": training,
        "evaluation_encounters": evaluation,
        "success_rate": successes / evaluation,
        "wells": len(agent.wells),
        "feedback_updates": agent.updates,
    }


if __name__ == "__main__":
    print(json.dumps([study(decays)[1] for decays in [(1.0,), (1.0, 0.9)]], indent=2))
