<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Franz X. Mohr
-->

## What this changes

<!-- One or two sentences. The diff says how; say why. -->

## Effect on the draws

<!--
The one question this project cannot answer from a diff, and the reason
CHANGELOG.md exists. Delete the lines that do not apply.

The `Sampler fingerprints` check builds base and head on one runner and reports
which fixtures moved, so the *fact* is automated -- but only you can say whether
it was intended.
-->

- [ ] **Unchanged** — no fingerprint moved. (Confirmed by the fingerprints check,
      or by a local before/after diff per CONTRIBUTING.md.)
- [ ] **Changed by a rounding error** — the arithmetic was reassociated, the
      posterior is the same. Worst relative difference:
- [ ] **Changed** — the posterior moved. Models and configurations affected, and
      why the new numbers are the right ones:
- [ ] **Not applicable** — this change cannot reach a sampler (docs, build,
      tooling).

## Checklist

- [ ] `ctest` passes locally
- [ ] `CHANGELOG.md` has an entry under *Unreleased* stating the effect on draws
- [ ] New code paths have a fixture in `test/CMakeLists.txt`, if a branch was
      added that nothing exercised before
- [ ] Anything vendored downstream is propagated, or an issue says it is pending
      — `src/core/` and `include/bayests/` are copied into the bvartools R
      package, which keeps its own release notes
