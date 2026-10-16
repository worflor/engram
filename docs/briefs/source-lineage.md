# Source lineage and import boundaries

This record identifies the standalone source-import decisions. The maintained [technical design](../TDD.md), [formats](../reference/FORMATS.md) and [capability map](../STATUS.md) define supported behavior; historical import decisions do not override those contracts.

## Source identity

The [provenance manifest](../../research/provenance.json) records import hashes and available source revisions. Portfolio `engram/` and `rag_tests/` sources were local ignored files, so a portfolio HEAD identifies checkout context rather than a revision of every imported file. Neuron supplies the native Rust baseline, including checked ENBR loading, fixtures and streaming continuity. The Python functional reference retains the richer memory and generation model.

## Repository boundaries

- Maintained Rust and Python implementations occupy the root source/module layout.
- Historical Rust and Python generations remain under `legacy/`; exploratory source and shaders remain under `research/`.
- The WDYM Java components are a consumer snapshot, excluding seed brains and external GloVe data.
- Source licenses and notices accompany the imported material.
- Corpora, working memories, caches and build outputs are excluded. The frozen Alexandria research artifact is documented separately.
- Consumer repositories are not migrated by an import into this repository. Registry publication remains disabled.

## Format reconciliation

The source generations disagreed about block starts, seeds, residual scales, compact RMS payloads and delta decoding. Compatibility readers retain their producer-specific layouts; the reference format defines an explicit boundary rather than assuming those byte streams were interchangeable. [FORMATS](../reference/FORMATS.md) describes V4 and supported historical reads.

Bidirectional decoder checks measure interoperability separately from source reconstruction error. The measured gallery retains difficult reconstruction cases so decoder agreement cannot be mistaken for an accuracy theorem. Original import hashes remain in provenance when maintained files receive corrections.

## Evidence scope

Import accounting establishes source lineage, not reproduction of every historical experiment. External-corpus studies, game integration and device checks require their own execution receipts. The [research index](../../research/README.md) separates those evidence domains from the maintained runtime contracts.
