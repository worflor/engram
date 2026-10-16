"""Retain a role-conditioned relation and project it without replaying history."""

import json
import numpy as np

from engram_relations import ContextRelations


def study():
    memory = ContextRelations(
        ("cue", "consequence"), representation="role-cue-outcome-v1"
    )
    for _ in range(8):
        for role, key in [(-1, "stranger"), (1, "owner")]:
            for cue in [-1.0, 1.0]:
                memory.push(key, [cue, role * cue])
    restored = ContextRelations.from_dict(json.loads(json.dumps(memory.to_dict())))
    owner, stranger = restored.moments("owner"), restored.moments("stranger")
    contrast = (owner.correlation[0, 1] - stranger.correlation[0, 1]) / 2
    assert np.isclose(contrast, 1)
    assert restored.moments("unfamiliar-role") is None
    # Normalization is a view. Constant consequences remain distinguishable in
    # the retained means even though their correlations are zero.
    levels = ContextRelations(
        ("cue", "consequence"), representation="constant-level-v1"
    )
    for cue in [-1.0, 1.0]:
        levels.push("safe", [cue, 1.0])
        levels.push("harmful", [cue, -1.0])
    safe, harmful = levels.moments("safe"), levels.moments("harmful")
    assert safe.correlation[0, 1] == harmful.correlation[0, 1] == 0
    assert safe.mean[1] != harmful.mean[1]
    return {
        "observations": owner.count + stranger.count,
        "owner_correlation": float(owner.correlation[0, 1]),
        "stranger_correlation": float(stranger.correlation[0, 1]),
        "conditional_contrast": float(contrast),
        "constant_consequence_means": [float(safe.mean[1]), float(harmful.mean[1])],
        "unknown_context": "absent, not silently assigned to a familiar role",
    }


if __name__ == "__main__":
    print(json.dumps(study(), indent=2))
    print(
        "Role keys and coordinate meanings are host-defined; the observed relationship is measured."
    )
