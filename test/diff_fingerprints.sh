#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr
#
# Compare two fingerprint recordings by fixture, and print only what moved.
#
#   test/diff_fingerprints.sh <before.txt> <after.txt>            # what moved
#   test/diff_fingerprints.sh <before.txt> <after.txt> <fixture>  # how it moved
#
# `diff before.txt after.txt` answers the question when the answer is "nothing":
# it prints nothing. When draws do move it prints every fingerprint line of
# every fixture that changed -- a full recording is close to a megabyte, and a
# change to a shared algorithm moves most of it -- which is far more than is
# needed to decide what to look at. This narrows it in two steps: the fixture
# names first, then one fixture's lines on request.
#
# Either argument may be a recording made by record_fingerprints.sh or a raw
# `ctest -V` redirect; both are reduced to the same fixture headers and
# fingerprint lines before anything is compared, so the two forms can even be
# compared against each other. Everything else a raw log carries -- progress
# bars, timings, absolute paths, the test numbering -- differs between two runs
# of an unchanged build and is dropped.
#
# A recording is then a sequence of blocks, each a fixture header ("<name>.h5
# [<Model>]") followed by its fingerprint lines. Blocks are keyed by fixture
# name and compared whole, header included, so a fixture that changed model type
# reads as moved rather than as unchanged.
#
# The `fingerprints.yml` workflow reports the same comparison, per fixture, for
# a pull request. It keeps its own implementation rather than calling this: it
# writes Markdown into the step summary, and it has to work when the base commit
# it checks out predates this file. Keep the two saying the same thing about the
# same recordings -- a fixture that reads as moved here should read as moved
# there.
#
# Exit status: 0 when every fixture matches, 1 when any moved or is present in
# only one of the recordings, 2 when the arguments are unusable. The second form
# exits 1 whenever the named fixture differs, which is the point of asking.

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <before.txt> <after.txt> [fixture]" >&2
    exit 2
fi

before=$1
after=$2
fixture=${3:-}

for f in "$before" "$after"; do
    if [ ! -f "$f" ]; then
        echo "$0: no such recording: $f" >&2
        exit 2
    fi
done

norm_a=$(mktemp)
norm_b=$(mktemp)
block_a=$(mktemp)
block_b=$(mktemp)
trap 'rm -f "$norm_a" "$norm_b" "$block_a" "$block_b"' EXIT

# Reduce a recording of either form to fixture headers and fingerprint lines.
# The two substitutions are what a raw log needs and an already-reduced file is
# unaffected by: keep only what follows the last carriage return, so a progress
# bar leaves nothing behind, and drop the `<n>: ` prefix `ctest -V` puts on
# every line of a test's output. What survives that is a header if it opens in
# column one and closes with the model type in brackets, and a fingerprint if it
# is an indented /posterior path. Anything else -- "Test command:", the timing
# lines, the summary -- matches neither and is dropped.
normalize() {
    awk '
        { sub(/.*\r/, ""); sub(/^[0-9]+: /, "") }
        /^[^ ].*\.h5 \[[^]]*\]$/ { print; next }
        /^  \/posterior\//       { print }
    ' "$1"
}

normalize "$before" >"$norm_a"
normalize "$after" >"$norm_b"

for pair in "$norm_a:$before" "$norm_b:$after"; do
    if [ ! -s "${pair%%:*}" ]; then
        echo "$0: no fingerprints found in ${pair#*:} -- is it a golden recording?" >&2
        exit 2
    fi
done

# Pulls one fixture's block -- header line and the indented fingerprint lines
# under it -- out of a normalized recording. A header is the only kind of line
# that starts in column one, which is what separates the blocks.
extract_block() {
    awk -v want="$2" '
        /^[^ ]/ { name = $1; sub(/\.h5$/, "", name); keep = (name == want) }
        keep    { print }
    ' "$1"
}

if [ -n "$fixture" ]; then
    extract_block "$norm_a" "$fixture" >"$block_a"
    extract_block "$norm_b" "$fixture" >"$block_b"

    if [ ! -s "$block_a" ] && [ ! -s "$block_b" ]; then
        echo "$0: no fixture named '$fixture' in either recording" >&2
        exit 2
    fi

    status=0
    diff -u --label "$before ($fixture)" --label "$after ($fixture)" \
         "$block_a" "$block_b" || status=$?
    if [ "$status" -eq 0 ]; then
        echo "$fixture unchanged"
    fi
    exit "$status"
fi

awk -v self="$0" -v aname="$before" -v bname="$after" '
    FNR == 1 { pass++; current = "" }

    # A fixture header opens a block. The header itself is part of what is
    # compared; the name it carries is the key.
    /^[^ ]/ {
        name = $1
        sub(/\.h5$/, "", name)
        if (pass == 1) {
            if (!(name in a)) { a_order[++na] = name }
            a[name] = a[name] $0 "\n"
        } else {
            if (!(name in b)) { b_order[++nb] = name }
            b[name] = b[name] $0 "\n"
        }
        current = name
        next
    }

    # An indented line belongs to the block above it. A recording that begins
    # with one is truncated at the front; count it rather than crediting it to
    # a fixture.
    {
        if (current == "") { orphan++; next }
        if (pass == 1) { a[current] = a[current] $0 "\n" }
        else           { b[current] = b[current] $0 "\n" }
    }

    END {
        for (i = 1; i <= na; i++) {
            name = a_order[i]
            if (name in b) {
                if (a[name] == b[name]) { same++ }
                else                    { moved[++nmoved] = name }
            } else {
                gone[++ngone] = name
            }
        }
        for (i = 1; i <= nb; i++) {
            if (!(b_order[i] in a)) { added[++nadded] = b_order[i] }
        }

        printf "%d fixtures in %s, %d in %s\n", na, aname, nb, bname
        printf "%d unchanged, %d moved\n", same, nmoved

        for (i = 1; i <= nmoved; i++) { printf "  moved   %s\n", moved[i] }
        for (i = 1; i <= ngone;  i++) { printf "  only in %s: %s\n", aname, gone[i] }
        for (i = 1; i <= nadded; i++) { printf "  only in %s: %s\n", bname, added[i] }

        if (orphan > 0) {
            printf "%d fingerprint lines before the first header - %s is truncated\n",
                   orphan, aname
        }

        if (nmoved + ngone + nadded == 0) { exit 0 }

        print ""
        print "Read one of them with:"
        printf "  %s %s %s %s\n", self, aname, bname,
               (nmoved ? moved[1] : (ngone ? gone[1] : added[1]))
        exit 1
    }
' "$norm_a" "$norm_b"
