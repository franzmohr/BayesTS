#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr
#
# The steps of .github/workflows/ci.yml, docs.yml and fingerprints.yml, run
# locally in the image docker/Dockerfile builds.
#
#   bayests-ci ci [Debug|Release]   build and test; Release also installs and packages
#   bayests-ci docs                 the Doxygen job, warnings surfaced
#   bayests-ci fingerprints [ref]   base against the working tree
#   bayests-ci shell                a prompt with the toolchain and /work ready
#
# The source is expected read-only at /src and is mirrored to /work, so nothing
# here can write into the repository -- the host tree of a Windows checkout
# carries a build/ directory full of MSYS2 objects and CMake caches that a Linux
# configure must not meet. Mount a named volume at /work to keep the mirror and
# the build tree between runs, which makes every run after the first incremental.

set -euo pipefail

SRC=/src
WORK=/work
OUT=/out

# Mirrors the workflow's ::group:: folding closely enough to read the same way.
group() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
err()   { printf '\033[31merror: %s\033[0m\n' "$*" >&2; }
die()   { err "$*"; exit 1; }

sync_source() {
    [ -d "$SRC" ] || die "mount the repository read-only at $SRC"
    [ -f "$SRC/CMakeLists.txt" ] || die "$SRC is not the BayesTS checkout: no CMakeLists.txt"

    group "Mirror $SRC to $WORK"
    # -a preserves timestamps, which is what lets ninja skip unchanged sources on
    # a second run; --delete keeps a file removed on the host from lingering here
    # and being compiled. .git comes along because the fingerprints job needs the
    # history; build/ and the generated fixtures do not, and the fixtures alone
    # run to well over a hundred megabytes.
    rsync -a --delete \
        --exclude=/build/ \
        --exclude=/test/baselines/ \
        --exclude='*.h5' \
        --exclude=CMakeUserPresets.json \
        "$SRC/" "$WORK/"
    cd "$WORK"
}

# The workflow spells its configure out rather than using a preset, because the
# presets carry no CMAKE_PREFIX_PATH -- that lives in the gitignored
# CMakeUserPresets.json. Same here, and for the same reason.
#
# BAYESTS_WERROR is on for the same reason it is on in the workflow, and it has
# to be on in both: this script exists so that what the Linux runner will say is
# known before the push, and a warning the runner turns into an error is exactly
# the kind of thing a Windows desktop does not see. The compiler here is the
# runner's, not the one on the host.
configure() {
    local build_dir=$1 build_type=$2
    shift 2
    cmake -S . -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_MAP_IMPORTED_CONFIG_DEBUG="Release;RelWithDebInfo;" \
        -DCMAKE_PREFIX_PATH="$BAYESTS_PREFIX_PATH" \
        -DHIGHFIVE_DIR="$BAYESTS_HIGHFIVE_DIR" \
        -DBAYESTS_NATIVE_ARCH=OFF \
        -DBAYESTS_WERROR=ON \
        "$@"
}

# ctest reports a failed golden test as "Failed" whether the binary threw or
# never started, and those want different fixes. Straight from the workflow.
diagnose() {
    local build_dir=$1
    local exe="$build_dir/test/bayests_golden"

    group "Diagnose: the binary and its shared libraries"
    ls -l "$exe" || { echo "the binary was never built"; return; }
    ldd "$exe" 2>&1 | grep -iE "not found|armadillo|hdf5" || ldd "$exe" 2>&1 | head -20

    group "Diagnose: does it start? (expect the usage line and exit 2)"
    "$exe" || echo "exit=$?"

    group "Diagnose: one fixture, threads pinned"
    # Any fixture but the grouped one, which is not at the root of its file and
    # so needs --group to say anything about the state of the build.
    local fixture
    fixture=$(find "$build_dir/test/fixtures" -name '*.h5' ! -name '*-grouped.h5' 2>/dev/null | head -1)
    if [ -n "$fixture" ]; then
        echo "using $fixture"
        "$exe" "$fixture" || echo "exit=$?"
    else
        echo "no fixture was generated -- the failure is before the samplers run"
    fi
}

job_ci() {
    local build_type=${1:-}
    if [ -z "$build_type" ]; then
        # The workflow's matrix is [Debug, Release]. It runs the two in parallel
        # on separate runners and this cannot, so both is the default here and
        # either one alone is an argument away.
        job_ci Debug
        job_ci Release
        return
    fi
    case "$build_type" in
        Debug|Release) ;;
        *) die "build type must be Debug or Release, not '$build_type'" ;;
    esac

    sync_source
    local lower=${build_type,,}
    local build_dir="build/bin/docker-$lower"

    group "Configure ($build_type)"
    # CMAKE_MAP_IMPORTED_CONFIG_DEBUG, set in configure() above, is what keeps
    # the Debug build honest: vcpkg installs a debug variant of every port beside
    # the release one, and a Debug build of this project would otherwise link
    # those -- testing BayesTS against a debug Armadillo and HDF5 that no user
    # has. Only BayesTS is built unoptimised, which is the point of the job.
    #
    # BAYESTS_TEST_PYTHON is ON rather than left at AUTO, as it is in the
    # workflow: the image carries python3-h5py, so a configure that does not
    # register the Python tests means something is wrong with the image, and the
    # AUTO default would report that as one line of status output and a suite
    # shorter than the runner's.
    configure "$build_dir" "$build_type" \
        -DBAYESTS_BUILD_DOCS=OFF \
        -DBAYESTS_BUILD_TESTS=ON \
        -DBAYESTS_TEST_PYTHON=ON \
        -DPython3_EXECUTABLE=/usr/bin/python3

    group "Build ($build_type)"
    cmake --build "$build_dir" --parallel

    group "Test ($build_type)"
    # Kept under /out either way: a green run is three lines worth reading and a
    # failing golden test prints the ~12 KB its fixture's run carries.
    if ! ctest --test-dir "$build_dir" --output-on-failure 2>&1 | tee "$OUT/ctest-$lower.log"; then
        diagnose "$build_dir"
        die "ctest failed ($build_type); the log is at $OUT/ctest-$lower.log"
    fi

    # The install and package steps are part of the contract too -- a missing
    # file in the install() list, or a CPack generator whose tool is absent, only
    # shows up here. Release only, as in the workflow.
    if [ "$build_type" = Release ]; then
        group "Install"
        cmake --install "$build_dir" --prefix "$RUNNER_TEMP/staged"

        group "Package"
        cmake --build "$build_dir" --target package
        # Copied by name rather than globbed by extension. A named volume keeps
        # the build tree between runs, so after a version bump the previous
        # version's archives are still sitting beside the new ones; a glob over
        # the extension alone carries those out too, restamped with this run's
        # time, and /out ends up describing two builds as though they were one.
        # CPackConfig.cmake holds the base name CPack just used -- written by
        # the configure, so it cannot disagree with the files on disk the way a
        # version parsed out of CMakeLists.txt could.
        local pkg_name
        pkg_name=$(sed -n 's/^set(CPACK_PACKAGE_FILE_NAME "\(.*\)")$/\1/p' \
                       "$build_dir/CPackConfig.cmake")
        [ -n "$pkg_name" ] \
            || die "no CPACK_PACKAGE_FILE_NAME in $build_dir/CPackConfig.cmake"
        find "$build_dir" -maxdepth 1 -name "$pkg_name.*" \
            -exec cp -v {} "$OUT/" \;
    fi

    group "$build_type: done"
}

job_docs() {
    sync_source
    local build_dir=build/bin/docker-docs

    group "Configure (docs)"
    # Tests off: this configure exists to produce a Doxyfile, and nothing in this
    # job compiles a sampler.
    configure "$build_dir" Release \
        -DBAYESTS_BUILD_DOCS=ON \
        -DBAYESTS_BUILD_TESTS=OFF

    group "Build docs"
    cmake --build "$build_dir" --target docs 2>&1 | tee "$OUT/doxygen.log"
    test -f "$build_dir/docs/html/index.html" || die "Doxygen produced no index.html"

    # Warnings are the point of running this, and are not fatal: Doxygen warns
    # about undocumented entities and this project documents deliberately rather
    # than exhaustively.
    if grep -E 'warning:' "$OUT/doxygen.log" > "$OUT/doxygen-warnings.txt"; then
        group "Doxygen warnings ($(wc -l < "$OUT/doxygen-warnings.txt"))"
        cat "$OUT/doxygen-warnings.txt"
    else
        group "No Doxygen warnings"
    fi

    rm -rf "$OUT/doxygen-html"
    cp -r "$build_dir/docs/html" "$OUT/doxygen-html"
    group "docs: done -- the site is at $OUT/doxygen-html"
}

job_fingerprints() {
    local base_ref=${1:-main}

    sync_source
    git config --global --add safe.directory "$WORK"
    git -C "$WORK" rev-parse --verify "$base_ref^{commit}" >/dev/null 2>&1 \
        || die "'$base_ref' does not resolve to a commit here; name the ref to compare against, e.g. 'fingerprints origin/main'"
    local base_sha
    base_sha=$(git -C "$WORK" rev-parse --short "$base_ref^{commit}")

    # Both recordings must be reduced by the *same* script, or the diff shows
    # differences in the extraction rather than in the samplers. Taking the
    # working tree's copy out first also makes this work when the base commit
    # predates the script.
    install -m 755 test/record_fingerprints.sh "$RUNNER_TEMP/record.sh"
    install -m 755 test/diff_fingerprints.sh "$RUNNER_TEMP/diff.sh"

    group "Record base ($base_ref, $base_sha)"
    # git archive rather than a checkout: the working tree is the other side of
    # this comparison, uncommitted changes included, and must not be disturbed.
    rm -rf /base
    mkdir -p /base
    git -C "$WORK" archive "$base_ref" | tar -x -C /base
    (
        cd /base
        configure build/bin/base Release -DBAYESTS_BUILD_DOCS=OFF -DBAYESTS_BUILD_TESTS=ON
        cmake --build build/bin/base --parallel
        "$RUNNER_TEMP/record.sh" build/bin/base "$RUNNER_TEMP/before.txt"
    )

    group "Record head (the working tree)"
    configure build/bin/docker-head Release -DBAYESTS_BUILD_DOCS=OFF -DBAYESTS_BUILD_TESTS=ON
    cmake --build build/bin/docker-head --parallel
    "$RUNNER_TEMP/record.sh" build/bin/docker-head "$RUNNER_TEMP/after.txt"

    cp "$RUNNER_TEMP/before.txt" "$RUNNER_TEMP/after.txt" "$OUT/"

    group "Compare"
    # The repository's own reducer, which names the fixtures that moved and
    # nothing else. The workflow prints that summary from an inline script; there
    # is no reason to keep a third copy of the logic here.
    "$RUNNER_TEMP/diff.sh" "$RUNNER_TEMP/before.txt" "$RUNNER_TEMP/after.txt" || true

    printf '%s\n' \
        "" \
        "Both recordings are under $OUT. To see the lines of one fixture that moved," \
        "name it as a third argument:" \
        "" \
        "    bayests-ci shell" \
        "    test/diff_fingerprints.sh /out/before.txt /out/after.txt <fixture>" \
        "" \
        "A moved number is often the point of the change. Whichever it is, CHANGELOG.md" \
        "has to say so -- see CONTRIBUTING.md, \"Recording the change\"."
}

usage() {
    cat <<'USAGE'
bayests-ci -- the Linux CI jobs, locally

  ci [Debug|Release]     .github/workflows/ci.yml, the linux job. Both build
                         types when neither is named.
  docs                   .github/workflows/docs.yml, the doxygen job.
  fingerprints [ref]     .github/workflows/fingerprints.yml: the given ref
                         (default main) against the working tree.
  shell                  bash, with the toolchain ready and /work in place.

Mounts: the repository read-only at /src, a directory to keep output in at /out,
and optionally a named volume at /work for incremental rebuilds.
USAGE
}

case "${1:-ci}" in
    ci)             shift || true; job_ci "$@" ;;
    docs)           shift || true; job_docs "$@" ;;
    fingerprints)   shift || true; job_fingerprints "$@" ;;
    shell|bash)     sync_source; exec bash ;;
    -h|--help|help) usage ;;
    *)              usage; die "unknown job '$1'" ;;
esac
