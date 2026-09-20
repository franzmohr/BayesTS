<!-- SPDX-License-Identifier: BSD-3-Clause -->
<!-- Copyright (c) 2026 Franz X. Mohr -->

# The Linux CI jobs, locally

The workflows in `.github/workflows/` run on Ubuntu, and the machine this
project is developed on is Windows. That gap is where a certain class of
breakage lives: a header MSYS2's GCC includes transitively and Ubuntu's does
not, a `Debug` build that only fails when it links a release Armadillo, an
`install()` list missing a file nobody notices until CPack runs. Finding those
normally means pushing and waiting.

This image is the same toolchain and the same dependency set as the Linux jobs,
baked once, so those jobs run here in the time a build takes.

Two files do the work:

- `Dockerfile` — Ubuntu 24.04, gfortran, an upstream CMake, Ninja, Armadillo and
  HDF5 from vcpkg's `x64-linux-dynamic` triplet, HighFive at the ref
  `.github/highfive-version` pins, Doxygen.
- `ci.sh` — the workflow steps, as a script short enough to read beside the YAML
  it mirrors. **When a workflow step changes, this is the file to change with
  it.**

## Build the image

From the repository root, not from here — the context is the root, which is what
lets the Dockerfile read the HighFive pin out of `.github/highfive-version`
instead of repeating it:

```bash
docker build -f docker/Dockerfile -t bayests-ci .
```

Ten to twenty minutes, nearly all of it vcpkg compiling Armadillo and HDF5 —
the step the workflow pays for on a cache miss. `.dockerignore` keeps the
context at about ten kilobytes; without it the generated fixtures alone would
send well over a hundred megabytes. Rebuild when `.github/highfive-version`
changes, or when the workflow's dependency set does.

**And rebuild after editing `ci.sh`.** It is copied into the image, not read
from the mounted checkout — the sources come through `/src`, the script does
not — so a change to it does nothing until the image is built again. That
rebuild is seconds rather than minutes: the `COPY` sits after vcpkg, so every
expensive layer is a cache hit.

## Run a job

From the repository root. The checkout goes in read-only, and anything worth
keeping comes back out through `/out`:

```powershell
docker run --rm -v "${PWD}:/src:ro" -v "${PWD}/build/docker-out:/out" bayests-ci ci Release
```

or from a POSIX shell, with `$PWD` in place of `${PWD}`:

```bash
docker run --rm -v "$PWD:/src:ro" -v "$PWD/build/docker-out:/out" bayests-ci ci Release
```

The jobs:

| Argument | Mirrors | Does |
| --- | --- | --- |
| `ci` | `ci.yml`, the `linux` job | Both build types, `Debug` then `Release` |
| `ci Release` | one matrix leg | Configure, build, `ctest`, install, package |
| `ci Debug` | the other | Configure, build, `ctest` |
| `docs` | `docs.yml`, the `doxygen` job | Builds the site, lists the warnings |
| `fingerprints [ref]` | `fingerprints.yml` | `ref` (default `main`) against the working tree |
| `shell` | — | A prompt, `/work` populated, the toolchain on `PATH` |

`ci` with no build type is the whole matrix, which the workflow runs as two
parallel runners and this runs back to back. `ci Release` is the one to reach
for while iterating.

Keep the build tree between runs with a named volume, and the second run only
compiles what changed:

```powershell
docker run --rm -v "${PWD}:/src:ro" -v "${PWD}/build/docker-out:/out" `
           -v bayests-ci-work:/work bayests-ci ci Release
```

## Before a push

`.githooks/pre-push` runs `ci` here on the commits `git push` is about to send
and refuses the push if `ctest` fails. Point git at the directory once per
clone — `core.hooksPath` is local configuration and cannot be committed:

```bash
git config core.hooksPath .githooks
```

It differs from the runs above in what it mounts at `/src`: the commit being
pushed, checked out into a throwaway worktree under `build/pre-push-src`, rather
than the working tree. That is what the runner will check out, and what a
pre-push check is for; the commands above answer the other question, whether
what is on disk right now would survive, and the hook says so when the tree is
dirty.

The whole matrix runs, `Debug` then `Release`, and the build tree lives in the
`bayests-ci-work` volume so that only the first push after a change pays for a
build. Four ways out, in increasing order of bluntness:

| | |
| --- | --- |
| `BAYESTS_PREPUSH_JOB="ci Release"` | one leg instead of both |
| `BAYESTS_PREPUSH_IMAGE`, `BAYESTS_PREPUSH_VOLUME` | a different image, or no volume when empty |
| `BAYESTS_PREPUSH=off` | skip the check |
| `git push --no-verify` | git's own bypass |

No docker, no running daemon or no image is a refusal naming the command that
fixes it rather than a pass. A check that quietly does nothing when its tooling
is missing is worse than no check, because it is still believed.

## What it does with the source

The sources are the one thing that does *not* come through the build context.
The checkout is mounted read-only at `/src` and mirrored with `rsync` to
`/work`, which is where the build happens, so nothing the container runs can
write into the repository — deliberately, because a Windows checkout's `build/`
directory holds MSYS2 objects and CMake caches that a Linux configure must not
meet, and the configure would poison them in return.

The mirror carries `.git` (the fingerprints job needs the history) and skips
`build/`, `test/baselines/` and every `*.h5` — the generated fixtures are well
over a hundred megabytes and the container regenerates them anyway.

Uncommitted changes are included, which is the point: this answers *will what I
have now survive CI*, a question the workflow cannot be asked until it is
pushed.

Output worth keeping — the `ctest` log, the packages, the Doxygen site, the
fingerprint recordings — is written to `/out`, which is why the run commands
above mount something there.

## Where it differs from the real thing

Worth knowing before reading a result as a verdict.

- **It is only the Linux half.** The `windows` job builds under MSYS2 UCRT64,
  which a local Windows build already covers.
- **Fingerprints are not comparable to a runner's.** They shift in the last
  digits with the compiler, the BLAS and the CPU, which is why the workflow
  builds both sides in one job and why none is checked in. A comparison made
  here is valid — both sides built here, one toolchain, one machine — but a
  number from it means nothing beside one from a runner.
- **`fingerprints` compares against a local ref**, not the merge result. The
  workflow measures what would land on the base branch; this measures the
  working tree as it stands. Closer to the procedure in `CONTRIBUTING.md` than
  to the workflow, and it uses the repository's own `record_fingerprints.sh` and
  `diff_fingerprints.sh` rather than the summary script embedded in the YAML.
- **CMake is pinned** at the `CMAKE_VERSION` in the Dockerfile, where the
  workflow takes whatever `lukka/get-cmake@latest` currently installs.
- **This is not `act`.** `act` replays the YAML, which here means re-running the
  vcpkg install on every invocation — the `actions/cache` step that saves the
  workflow from that has nothing behind it locally. The expensive part of these
  workflows is the dependency tree rather than the YAML, so the tree is baked
  into a layer and the steps are reimplemented in `ci.sh`.
- **amd64 only.** The vcpkg triplet is `x64-linux-dynamic`, matching the runner.
  On an arm64 host this works under emulation, slowly.
