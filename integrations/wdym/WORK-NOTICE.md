# What Do You Mean? license

Copyright (c) 2026 Michael Bickford ("Woflo"), publishing as Woflo Labs.

Permission is granted to copy and distribute this Work Notice verbatim with
What Do You Mean? (WDYM). A modified Work Notice may accompany a permitted
modified version of WDYM, but it must accurately identify the licenses and
applicable notices and must not imply endorsement by Michael Bickford or Woflo
Labs.

WDYM is distributed under mixed terms. Most of the project is GPL; the reusable
Engram implementation identified below uses the Woflo Labs Community Source
License. This file is the controlling Work Notice and the exact boundary between
them.

**Work:** What Do You Mean?, including the original source code, documentation,
tests, fixtures, assets, and Engram components in this repository.

**Source distribution:** A complete corresponding-source archive is attached to each
public WDYM release. The development repository is private.

**Licensor:** Michael Bickford ("Woflo"), publishing as Woflo Labs.

## Most of WDYM: GPL-3.0-or-later

Unless a path is listed in the next section or carries its own third-party
notice, original material in this repository is licensed under the GNU General
Public License, version 3 or later, with the WDYM-Woflo Engram Components
Exception.

The legal instruments in this `LICENSE.md` and in `LICENSES/**` are not project
source or documentation covered by a project-code license. Each is governed by
the copying permission stated in its own text.

- [GNU GPL version 3](LICENSES/GPL-3.0-or-later.txt)
- [WDYM-Woflo Engram Components Exception 1.0](LICENSES/WDYM-WOFLO-ENGRAM-EXCEPTION-1.0.md)

WDYM-specific code that calls, displays, tests, or integrates a protected Engram
component remains GPL-covered unless its own path is listed below. This includes
the UI, configuration surfaces, context integration, prompts, and test suites
that exercise Engram behavior.

## Engram components: WLCSL-1.0

These paths are licensed under the
[Woflo Labs Community Source License 1.0](LICENSES/WLCSL-1.0.md):

- `src/main/java/woflo/wdym/engram/**`
- `src/main/resources/engram/seed.engram`

The `**` entry covers that directory tree. Covered source files whose format
allows comments also carry a `LicenseRef-WLCSL-1.0` notice. This list controls.

The GloVe-derived vocabulary at
`src/main/resources/engram/glove_20k_300d.bin` is third-party material under its
own terms. It is not relicensed as an Engram component. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Combined WDYM builds

The WDYM-Woflo exception permits the GPL-covered parts of WDYM to be linked,
combined, and distributed with the WLCSL-covered Engram components. Each part
keeps its own license, and a combined distribution must satisfy the GPL, the
exception, and the applicable Engram terms.

The exception permits the combination; it does not convert the Engram components
to GPL or waive their conditions. A GPL-only fork may remove the protected
components and the code paths that require them, but this notice does not promise
that such a configuration already compiles unchanged.

## Contributions

Contributions accepted after this notice follow the
[Woflo Labs Contributor Agreement 1.0](LICENSES/CONTRIBUTOR-AGREEMENT-1.0.md)
and the acceptance process in [CONTRIBUTING.md](CONTRIBUTING.md).

A contribution accepted into a GPL-covered path is promised back publicly under
GPL-3.0-or-later with the WDYM-Woflo exception. A contribution accepted into a
protected Engram path is promised back publicly under WLCSL-1.0.

## Patents, third-party material, and names

Some Engram components may implement subject matter described in pending patent
applications. WLCSL-1.0 grants patent rights only for the uses stated there.
"Patent pending" does not mean that a patent has issued or that any particular
claim will be allowed.

Dependencies, referenced projects, vendor material, and files carrying their own
notices are not relicensed here; their own terms continue to apply. Bundled and
referenced third-party material is inventoried in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Neither license grants trademark rights in Woflo Labs, What Do You Mean?, WDYM,
The Doctor, Engram, Whisper, Alexandria, or related names and visual identities,
except the descriptive attribution allowed by the applicable license.

Woflo Labs is Michael Bickford's publishing and independent research identity,
not a separate incorporated entity as of this notice.

## Prior releases

This Work Notice governs versions distributed with it. It does not revoke or
narrow rights that recipients may have received under the license accompanying an
earlier distribution.
