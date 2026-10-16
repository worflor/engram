# Consumer integrations

## Maintained Minecraft NPC laboratory

`minecraft/` is a pinned development-only Fabric/GameTest adapter for the canonical Python experience NPC. The file lane controls a real villager and captures native world observations and action consequences. It is independent of the historical Java snapshot below. [Reproduction, ownership and behavioral checks](../docs/reference/MINECRAFT-NPC.md), [game-independent NPC API](../docs/reference/NPC.md).

## Neuron

The root Rust crate was extracted from `worflor/neuron` at the revision in `research/provenance.json`. Neuron uses the mathematical library within its gesture/dynamics systems. Its hardware, input dispatch and application code remain in Neuron. No consumer dependency was changed by this extraction.

## What Do You Mean?

`wdym/java/woflo/wdym/engram/` preserves `EngramBrain`, `EngramSegment`, `EngramService` and `GloveEmbedder` from the capstone's current local repository. This is a host integration snapshot, not a standalone Java package. `EngramService` depends on Minecraft and host classes; the memory implementation uses Gson.

No seed brain, player context, model weights or third-party embedding binary is copied. The original WDYM Work Notice and WLCSL text are retained beside the snapshot. Only its listed Engram source components are extracted; the rest of WDYM is outside this repository.

## Manifold

[Manifold](https://github.com/worflor/git-desktop-premium-ultra-promax-plus-R) uses Engram-derived K/G features alongside identifier overlap to weight relationships between code hunks. Its bootstrap loads the bundled Alexandria memory and a separate embedding vocabulary; its hunk encoder fits trajectories in that representation.

The frozen original Alexandria ZIP behind Manifold's bundled ENDB export is included as a [research artifact](../research/README.md#alexandria-research-artifact). Manifold's Dart implementation and exporter remain in that consumer repository. No Manifold process, embedding vocabulary or application code is copied for this addition.

## Integrating from here

Rust callers can use a path or Git dependency and the existing module names. Python callers install `woflo-engram` and import `engram_codec`. Pin a commit for reproducible consumers. Compare packets only in a known coordinate basis. Check source-to-decoded error on the application's actual workload before setting tolerances or making recognition decisions.

Full v3 packet output is a deliberate compatibility boundary. Adopting it in a consumer requires that consumer's tests and migration decision. Python ZIP memory and Rust ENBR snapshots require separate readers.
