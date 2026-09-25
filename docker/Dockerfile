# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

# The Linux half of .github/workflows/ci.yml, as an image -- docs.yml and
# fingerprints.yml along with it.
#
# What it is for: the Linux jobs are the ones a Windows desktop cannot run, so a
# broken build or a fixture that throws under gcc is normally found by pushing
# and waiting. This is the same toolchain and the same dependency set, baked
# once, so the same jobs run locally in the time a build takes.
#
# The build context is the repository root -- `docker build -f docker/Dockerfile
# .` -- which is what lets the HighFive pin below be COPYied out of
# .github/highfive-version rather than repeated here as a version that later
# drifts out of step with the workflows. The *sources* still arrive at run time
# through a read-only bind mount at /src rather than through the context; see
# ci.sh for why they must not be built where they lie.
#
# It is *not* nektos/act. act replays the YAML, which here means re-running the
# vcpkg install on every invocation (the actions/cache step it would honour does
# not exist locally) and providing a JavaScript runtime for actions/checkout. The
# expensive part of these workflows is the dependency tree, not the YAML, so this
# bakes the tree into a layer and reimplements the steps in ci.sh, which is short
# enough to read beside the workflow it mirrors. When a workflow step changes,
# that script is the thing to change with it.
#
# amd64 only, deliberately: the vcpkg triplet below is x64, matching the runner.
# On an arm64 host this builds and runs under emulation, slowly.

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

# Everything apt provides. gfortran is not optional -- the top-level project()
# declares C, CXX and Fortran -- and it is also what vcpkg's openblas needs to
# build. rsync is how ci.sh mirrors the read-only source mount into the build
# tree while preserving timestamps, which is what keeps a second run incremental.
# doxygen and graphviz are here for the docs job; they cost little and having one
# image rather than two keeps the two jobs honest about using the same compiler.
#
# python3-h5py and python3-numpy are what the Python tests need, and ci.sh asks
# for them by name rather than letting BAYESTS_TEST_PYTHON=AUTO decide.
# Without them the image ran fewer tests than the Linux runner does and said
# so in a line nobody reads, which is the opposite of what a pre-push gate is
# for. apt rather than pip, as in the workflow: the system Python is externally
# managed and the distribution's h5py is all the tests need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        curl \
        ca-certificates \
        zip \
        unzip \
        tar \
        pkg-config \
        ninja-build \
        python3 \
        python3-h5py \
        python3-numpy \
        rsync \
        doxygen \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# CMake from upstream, not apt. cmake_minimum_required is 3.25 and Ubuntu ships
# less; the workflow solves this with lukka/get-cmake, which installs the latest
# upstream release. Pinned here instead of tracking latest, because an image is
# rebuilt rarely and a version that moves under it turns an unrelated failure
# into a mystery. Override with --build-arg CMAKE_VERSION=... to test a newer one.
ARG CMAKE_VERSION=3.31.6
RUN curl -fsSL -o /tmp/cmake.tar.gz \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    && mkdir -p /opt/cmake \
    && tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1 \
    && rm /tmp/cmake.tar.gz \
    && ln -s /opt/cmake/bin/cmake  /usr/local/bin/cmake \
    && ln -s /opt/cmake/bin/ctest  /usr/local/bin/ctest \
    && ln -s /opt/cmake/bin/cpack  /usr/local/bin/cpack \
    && cmake --version | head -1

# Armadillo and HDF5 from vcpkg, for the reason .github/actions/setup-linux-deps
# gives at length: the project finds both in CONFIG mode and links `armadillo` and
# `hdf5::hdf5-shared`, target names only the upstream CMake configs define.
# Debian's packages ship neither. The triplet has to be the dynamic one -- the
# default x64-linux is static-only and would offer hdf5::hdf5-static instead.
#
# This layer is the whole point of the image: it is the ten-odd minutes the
# workflow spends on a cache miss, spent once here instead of once per run.
ARG VCPKG_TRIPLET=x64-linux-dynamic
ENV VCPKG_TRIPLET=${VCPKG_TRIPLET}
RUN git clone --depth 1 https://github.com/microsoft/vcpkg /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics \
    && /opt/vcpkg/vcpkg install --triplet "${VCPKG_TRIPLET}" \
         --x-install-root=/opt/vcpkg-installed armadillo hdf5 \
    # The installed tree is all CMake reads. Everything else vcpkg leaves behind
    # is intermediate and roughly doubles the image.
    && rm -rf /opt/vcpkg/buildtrees /opt/vcpkg/downloads /opt/vcpkg/packages \
              /root/.cache/vcpkg

# Fail at build time rather than at find_package, whose message names a missing
# .cmake file and says nothing about how it was meant to get there.
RUN set -eu; \
    prefix="/opt/vcpkg-installed/${VCPKG_TRIPLET}"; \
    for pkg in armadillo hdf5; do \
        found=$(find "$prefix/share" -maxdepth 2 -iname '*config.cmake' -ipath "*$pkg*" | head -3); \
        [ -n "$found" ] || { echo "no CMake config for $pkg under $prefix" >&2; exit 1; }; \
        echo "$pkg: $found"; \
    done

# HighFive, header-only, at the ref .github/highfive-version pins -- the same
# file the workflows read, copied in so the image cannot drift from them. That
# single pin is the reason this Dockerfile takes the repository root as its
# context rather than docker/.
#
# Note the version numbering trap recorded in the composite action: 3.0.0 exists
# only as v3.0.0-beta1 and v3.0.0-beta2 while the beta's header already reports
# 3.0.0, so the tag cannot be read off H5Version.hpp.
COPY .github/highfive-version /tmp/highfive-version
RUN ref=$(tr -d '[:space:]' < /tmp/highfive-version) \
    && echo "HighFive ref: $ref" \
    && git clone --depth 1 --branch "$ref" \
         https://github.com/BlueBrain/HighFive.git /opt/highfive \
    && test -f /opt/highfive/include/highfive/H5File.hpp \
    && rm -rf /opt/highfive/.git /tmp/highfive-version

# The two the golden harness refuses to run without. A multi-threaded BLAS
# reassociates floating-point reductions, and a fingerprint that cannot be
# reproduced is worse than none. test/CMakeLists.txt sets them per test; setting
# them here covers the fixture generators and anything run by hand in a shell.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Where ci.sh looks for the things the workflow passes as step outputs.
ENV BAYESTS_PREFIX_PATH=/opt/vcpkg-installed/${VCPKG_TRIPLET} \
    BAYESTS_HIGHFIVE_DIR=/opt/highfive \
    RUNNER_TEMP=/tmp/runner

COPY docker/ci.sh /usr/local/bin/bayests-ci
RUN chmod +x /usr/local/bin/bayests-ci && mkdir -p /work /out "$RUNNER_TEMP"

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/bayests-ci"]
CMD ["ci"]
