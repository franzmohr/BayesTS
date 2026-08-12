#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr
#
# Record the golden fingerprints in a form that can be diffed.
#
#   test/record_fingerprints.sh <binaryDir> <out.txt>
#
# `ctest -V` interleaves the fingerprints with progress bars, timings and
# absolute paths, all of which differ between two runs of an unchanged build.
# This keeps the fixture headers and the fingerprint lines and nothing else, so
# a diff of two recordings shows exactly the sampler output that moved:
#
#   test/record_fingerprints.sh build/bin/<preset> before.txt
#   ...make the change, rebuild...
#   test/record_fingerprints.sh build/bin/<preset> after.txt
#   diff before.txt after.txt
#
# Fingerprints shift in the last digits with the compiler, the BLAS and the CPU,
# so the two recordings have to come from the same machine and the same build
# type. See test/CMakeLists.txt for why none of this is checked in.

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <binaryDir> <out.txt>" >&2
    exit 2
fi

# ctest exits non-zero if a fixture throws; report that rather than a diff of a
# truncated recording. `tee` would mask it, hence the temporary file.
raw=$(mktemp)
trap 'rm -f "$raw"' EXIT

status=0
ctest --test-dir "$1" -V >"$raw" 2>&1 || status=$?

# The fixture header ("<name>.h5 [<Model>]") and the per-dataset fingerprints
# ("  /posterior/... " or "  ... absent"). Progress bars are stripped by taking
# only the part of the line after the last carriage return.
sed 's/.*\r//' "$raw" \
    | sed -n 's/^[0-9]*: \(.*\.h5 \[.*\]\)$/\1/p; s/^[0-9]*:   \(\/posterior\/.*\)$/  \1/p' \
    >"$2"

if [ "$status" -ne 0 ]; then
    echo "ctest exited $status -- recording in $2 may be incomplete" >&2
    exit "$status"
fi

echo "Recorded $(grep -c '^  /posterior/' "$2") fingerprints to $2"
