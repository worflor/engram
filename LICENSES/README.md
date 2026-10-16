# License boundaries

This extraction preserves the licenses attached to its sources. It does not adopt Vixen's MIT license.

- Root `src/`, Rust tests/fixtures and their derivatives retain the Engram Work Notice and WLCSL-1.0 reproduced in `../LICENSE.md` from Neuron.
- `engram_codec.py`, portfolio research, legacy source and their derivatives retain the portfolio's `Whisper-Protocol-1.0.md` in this directory. Python package metadata points to that text.
- `integrations/wdym/` retains the WDYM Work Notice and its Engram license text. The extracted Java paths are the WLCSL-covered paths listed in that notice.
- `research/artifacts/alexandria.engram` preserves the original ZIP content corresponding to Manifold's WLCSL-1.0 Alexandria research component. Its exact source identity and payload match are recorded in `research/provenance.json`; this addition does not include the separate embedding vocabulary.
- New standalone documentation, tools, examples and instrument use WLCSL-1.0 under the root Engram Work Notice, except modifications to portfolio-derived code, which retain its original notice above.
- The new reference modules `engram_wire.py`, `engram_analysis.py`, `engram_memory_io.py` and `engram_embeddings.py` use the root WLCSL-1.0 notice. The Python distribution includes both license texts and declares their combined coverage; the portfolio-derived facade retains its original terms.
- Dependencies and any separately noticed third-party material retain their own terms. No third-party corpus or embedding dataset is included.
- The Minecraft lane's new source uses the root Engram Work Notice. Its unchanged official Gradle wrapper retains Apache-2.0; the accompanying Gradle license/notice files identify that third-party component. Minecraft, Fabric and downloaded dependencies retain their own licenses and are not vendored into the reference package.
- `engram_kernels/fit.wgsl` is the portfolio-derived fitting kernel and retains the Whisper Protocol License, alongside its historical source. New learning/training/retrieval/metrics/acceleration wrappers, examples and lab use the root WLCSL-1.0 notice; modifications to the portfolio-derived facade retain its original coverage.

Source identities and hashes are recorded in `research/provenance.json`. Repository visibility and registry publication are separate choices from licensing.
