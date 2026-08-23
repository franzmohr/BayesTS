# BayesTS

[![CI](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml)
[![Docs](https://github.com/franzmohr/BayesTS/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/franzmohr/BayesTS/actions/workflows/docs.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A5%203.25-blue.svg)](CMakeLists.txt)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml)

Bayesian time series estimation in C++: Gibbs samplers for vector autoregressive
models, driven either from the command line or from an embedded host such as an
R package.

A *model file* — one HDF5 file holding the data, the priors, the starting values
and a `/model/algorithm` name — is the unit of work. `bayests` reads it, runs the
sampler that name selects, and writes the posterior draws, forecasts and
pointwise log likelihood back into the same file. Nothing is passed on the
command line except which file to work on and which of those three results are
wanted, so a run is fully described by the file it is given and can be repeated
from it.

The numerics are deliberately isolated. `bayests_core` links neither HDF5 nor
HighFive, prints nothing, and touches no global state beyond the Armadillo RNG:
values in, values out. That is what lets the same sampler objects serve this
command line and a host that cannot allow a library to own the files or the
console — under RcppArmadillo the RNG is R's own, so `set.seed()` reaches the
draws without the sampler knowing. Progress reporting and cancellation arrive
through a `Reporter` interface instead of being written to `stdout`.

Requires a C++20 compiler and Fortran (for LAPACK). Built on
[Armadillo](https://arma.sourceforge.net/), [OpenBLAS/LAPACK](https://www.openblas.net/),
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) and
[HighFive](https://github.com/BlueBrain/HighFive), with optional OpenMP.

## Models

The `/model/algorithm` attribute in the model file selects the sampler; it is
the only thing that decides which one runs. Six are registered:

| `algorithm` | Coefficients | Error precision | Variable selection |
| --- | --- | --- | --- |
| `VarNormalWishart` | Constant, normal prior | Wishart | SSVS, BVS |
| `VarNormalGamma` | Constant, normal prior | Independent gamma, optional constant covariance block | SSVS, BVS |
| `VarNormalStochvol` | Constant, normal prior | Stochastic volatility, optional covariance block | BVS |
| `VarTvpGamma` | Random walk | Independent gamma, optional time-varying covariance block | BVS |
| `VarTvpWishart` | Random walk | Wishart | BVS |
| `VarTvpStochvol` | Random walk | Stochastic volatility, optional time-varying covariance block | BVS |

Every model supports exogenous regressors, deterministic terms, a structural
(contemporaneous-coefficient) form, forecasting and a pointwise log likelihood
laid out for WAIC and PSIS-LOO.

Two algorithms carry the implementation weight. The time-varying coefficient
paths are drawn as a single block with the simulation smoother of Durbin and
Koopman (2002), so the whole path moves at once rather than period by period.
Stochastic volatility uses the ten-component normal mixture of Omori et al.
(2007), which turns the non-linear measurement equation into a conditionally
linear one. `VarTvpStochvol` combines both.

SSVS is available only for the two constant-coefficient models with a gamma or
Wishart error precision; the stochastic volatility and time-varying parameter
samplers reject it rather than silently ignoring it.

## Usage

```
bayests <command> <path> [flags...]
```

`<path>` is either a single HDF5 model file or a directory, which is walked
recursively and every HDF5 file in it processed. Results are written back into
each file in place.

| Command | What it computes |
| --- | --- |
| `posterior` | All three of the below, in the order coefficients, log likelihood, forecasts |
| `coefficients` | Posterior draws of the coefficients and the error precision |
| `forecasts` | One forecast path per posterior draw |
| `loglik` | Pointwise log likelihood, draws × periods |

`coefficients`, `forecasts` and `loglik` take no flags. `posterior` runs all
three by default and any of them can be switched off:

```bash
# Everything
bayests posterior model.h5

# Draw the coefficients, but skip the forecast and the log likelihood
bayests posterior model.h5 --no-forecasts --no-loglik

# Every model file below models/, forecasts only
bayests forecasts models/
```

`forecasts` and `loglik` read coefficient draws that are already in the file, so
they are the way to add a result to a model that has been sampled once, without
re-running the sampler.

The program reports its thread counts on startup and exits 1 on a bad path, a
non-HDF5 file, or an unknown `algorithm`. In directory mode a file that fails is
reported on `stderr` and the walk continues to the next one, but the exit status
is 1 if any file failed — so a script driving a directory of models can tell
whether everything in it was processed.

### The model file

Groups and datasets the readers look for. Only `/model`'s `algorithm`, `k`,
`iterations` and `burnin` are required — everything else is read through a
default, so a file written for a simpler model still describes a valid one.

| Location | Contents |
| --- | --- |
| `/model` (attributes) | `algorithm`, `k` endogenous variables, `iterations` kept, `burnin` discarded; optional `p`, `m`, `s`, `n` (lags, exogenous variables, their lags, deterministic terms), `h` forecast horizon, `varsel` (`none`, `ssvs`, `bvs`), `structural`, `error` |
| `/data/train/y`, `/data/train/z` | Endogenous variables and the regressor matrix |
| `/data/forecast/z` | Out-of-sample regressors; required when `h` > 0 |
| `/priors/a`, `/priors/psi` | Normal prior `mu` and `v_inv` for the coefficients and the covariance block, plus `inprior`, `include`, and `tau0`/`tau1` for SSVS |
| `/priors/u_sigma` | `shape`/`rate` for gamma precisions, `df`/`scale` for Wishart, `mu`/`v_inv`/`sigma`/`offset` for stochastic volatility |
| `/initial/…` | Starting values: `a`, `psi`, `u_sigma_inv`, `u_omega_inv`, `h`, the `*_init` states and the `*_lambda`, `*_sigma_inv` blocks the samplers that need them read |
| `/posterior/…` | Written by the run: `a/coeffs`, `a/lambda`, `a/sigma`, the matching `psi/…`, `u_sigma_inv/coeffs`, `u_omega_inv/coeffs`, `forecast` and `loglik` |

Two conventions are worth knowing before writing a file by hand. Draws run
along the **rows** on disk, which is what R and `coda` expect of an `mcmc`
object, and are transposed to one draw per column inside the samplers.
Variable-selection positions are stored **one-based**, the way R and the file
format count, and converted on read. The `error` attribute is what turns the
covariance block on, and its spelling is model-specific: `gamma+covar` for the
gamma models, `sv+covar` for stochastic volatility, `wishart` for
`VarTvpWishart`.

## Building from source

The build needs nothing but CMake and the four dependencies above. The toolchain
notes below are for Windows, which is the platform the project is developed and
exercised on; the runtime-bundling step in `CMakeLists.txt` is Windows-specific,
but nothing else is.

### Windows

**Compiler**

On my Windows set-up I use the `x64-mingw-dynamic` compiler, which I downloaded via MSYS2 using:

`pacman -S --needed base-devel mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-toolchain`

Make sure you also install gfortran, which is needed for LAPACK.

[https://code.visualstudio.com/docs/cpp/config-mingw] is an informative guide.

**Visual Studio Code**

Should not require any further tweaking, if the CMakePresets.json is configured to your system.

**Ninja**

Using Ninja as generator. Make sure that its directory is included in the environment variable `Path`.

**vcpkg for package management**

Make sure that its directory is included in the environment variable `Path`. Also add the environment variable `VCPKG_ROOT`, which should also include the directory.

Note that the `hdf5` library has problems to build. So I installed the binary from the website.

General information: [https://learn.microsoft.com/en-us/vcpkg/] and [https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-vscode?pivots=shell-powershell]

**CMake 4.2.1**

Make sure that its directory is included in the environment variable `Path`.

`CMakePresets.json` is committed and should stay machine-independent. Put your own
compiler and dependency paths in a `CMakeUserPresets.json` next to it, which is
git-ignored, and inherit from the `default` preset:

```json
{
    "version": 8,
    "configurePresets": [
        {
            "name": "my-windows-default",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_C_COMPILER": "gcc",
                "CMAKE_CXX_COMPILER": "g++",
                "CMAKE_Fortran_COMPILER": "gfortran",
                "CMAKE_PREFIX_PATH": "D:/vcpkg/installed/x64-mingw-dynamic;D:/HDF5/2.0.0;D:/HighFive",
                "HIGHFIVE_DIR": "D:/HighFive",
                "VCPKG_HOST_TRIPLET": "x64-mingw-dynamic",
                "VCPKG_TARGET_TRIPLET": "x64-mingw-dynamic"
            }
        }
    ]
}
```

Things you will want to adjust:
* The compiler variables. No change needed if you also use `gcc` and `g++`.
* `VCPKG_HOST_TRIPLET` and `VCPKG_TARGET_TRIPLET`, which seem to be `x64-windows` by default and, thus, assume that you have Visual Studio (not Visual Studio Code) installed.
* `CMAKE_PREFIX_PATH`, so that Armadillo, HDF5 and HighFive are found where `vcpkg` (or the installer) put them.

**Doxygen (optional, for the API documentation)**

Only needed to build the `docs` target. Install it from
[https://www.doxygen.nl/download.html] or, under MSYS2, with

`pacman -S mingw-w64-x86_64-doxygen mingw-w64-x86_64-graphviz`

Graphviz is optional and only adds the class and include graphs. Make sure both
directories are on `Path`, or point CMake at the installation with
`-DDOXYGEN_EXECUTABLE=...`. See the *API documentation* section below.

### Linux

Everything comes from the distribution's own packages; no vcpkg is needed. On
Debian or Ubuntu:

```bash
sudo apt install cmake ninja-build gfortran libarmadillo-dev libhdf5-dev libopenblas-dev
```

HighFive is the one exception. The archive is still on the 2.x line and this
project is written against 3.x, so clone the tag recorded in
`.github/highfive-version` and point the build at it — it is header-only, so
there is nothing to install:

```bash
git clone --depth 1 --branch "$(cat .github/highfive-version)" \
    https://github.com/BlueBrain/HighFive.git ~/src/highfive

cmake -S . -B build/bin/linux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIGHFIVE_DIR="$HOME/src/highfive" \
    -DBAYESTS_BUILD_DOCS=OFF
cmake --build build/bin/linux
```

Neither `libarmadillo-dev` nor `libhdf5-dev` installs a CMake config package, so
configuration falls back to `FindArmadillo` and `FindHDF5` and builds the
imported targets from what those report. The configure output says which route
each dependency took:

```
-- Armadillo: FindArmadillo module, version 12.6.7, libraries /usr/lib/x86_64-linux-gnu/libarmadillo.so
-- HDF5 target: HDF5::HDF5
```

A `-- Armadillo: CONFIG package` line instead means a config package was found —
from vcpkg, or a source build — and that is used in preference.

### Configure and build

```bash
# Configure -- run from the repository root
cmake --preset my-windows-default

# Build
cmake --build build/bin/my-windows-default
```

**Build tree location**

The presets set `binaryDir` to `${sourceDir}/build/bin/${presetName}`, so each
one gets its own tree under `build/` and they do not overwrite each other.
`build/` is git-ignored and excluded from the source package.

Configuring in the repository root itself is refused with a `FATAL_ERROR`: an
in-source build scatters `CMakeCache.txt` and `CMakeFiles/` among the sources,
where they shadow every later out-of-source configure. Point `-B` at a
subdirectory, or use a preset, which already does.

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `BAYESTS_BUILD_DOCS` | `ON` | Add the `docs` target when Doxygen is found |
| `BAYESTS_BUILD_TESTS` | `ON` | Build the regression harnesses in `test/` |
| `BAYESTS_NATIVE_ARCH` | `OFF` | `-march=native`; not redistributable, see *Packaging* |
| `BAYESTS_BUNDLE_RUNTIME_DEPS` | `ON` (Windows) | Copy the runtime DLLs next to the executable |
| `BAYESTS_WISHART_FIXTURE` | *(empty)* | Recorded model file the `VarNormalWishart` tests derive from; those tests are skipped when unset |

## Tests

`ctest` is the entry point, and needs no arguments or environment beyond a
completed build:

```bash
cmake --build build/bin/my-windows-default
ctest --test-dir build/bin/my-windows-default --output-on-failure
```

Each `fixture.*` test writes one model file into
`build/bin/<preset>/test/fixtures/`, and the `golden.*` test after it runs all
three entry points over that file. They are paired through CTest's
`FIXTURES_SETUP`/`FIXTURES_REQUIRED`, so naming a single golden test regenerates
just the input it needs:

```bash
ctest --test-dir build/bin/my-windows-default -R golden.VarTvpGamma-covar
```

Fixtures are written into the build tree, never into the source tree: they are
derived data and a full set runs to hundreds of megabytes. The tests pin
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` themselves, because the
samplers are only reproducible single-threaded, and on Windows they prepend the
dependency DLL directories to `PATH` so `ctest` works in a shell that has not
been set up by hand.

**What these tests do and do not check.** `bayests_golden` prints a fingerprint
for every posterior dataset and exits non-zero only if a fixture throws, so the
suite is a smoke test across every sampler and code path rather than a numerical
regression test. Expected fingerprints are deliberately not checked in: they
shift in the last digits with the compiler, the BLAS implementation and — when
`BAYESTS_NATIVE_ARCH` is on — the CPU, so a committed baseline would fail on
every machine but the one that recorded it. To use them as a regression check,
record the output before a change and diff it against the output after, on the
same machine and with the same build:

```bash
ctest --test-dir build/bin/my-windows-default -R golden -V > test/baselines/before.txt
# ... make the change, rebuild ...
ctest --test-dir build/bin/my-windows-default -R golden -V > test/baselines/after.txt
diff test/baselines/before.txt test/baselines/after.txt
```

`test/baselines/` is the place for these: it is git-ignored except for its own
`.gitignore`, and unlike anything under `build/` it survives a `rm -rf build`,
which a baseline recorded before a rebuild has to. Date the files or name them
for the change they precede — a baseline recorded before a *build flag* change
still diffs cleanly enough to look meaningful, which makes a stale one worse
than none.

**Coverage.** Five of the six samplers are covered from a clean clone —
`VarNormalGamma`, `VarNormalStochvol`, `VarTvpGamma`, `VarTvpWishart` and
`VarTvpStochvol` — which are the five `test/make_model_fixture.cpp` can write
from scratch.

`VarNormalWishart` is the exception: it cannot be generated, only derived from a
recorded model file with `make_varsel_fixture`, so its three tests appear only
when `BAYESTS_WISHART_FIXTURE` points at one. Teaching
`test/make_model_fixture.cpp` to emit it would remove the last dependency on
data that is not in the repository.

## Packaging

Packages are produced with CPack from the `release` preset. Like the
development presets, `release` carries no compiler or dependency paths of its
own — inherit from it in your `CMakeUserPresets.json`, listing it *before* your
own preset so its `Release` build type wins:

```json
{
    "name": "my-windows-release",
    "inherits": ["release", "my-windows-default"]
}
```

Then configure, build and package:

```bash
cmake --preset my-windows-release
cmake --build --preset my-windows-release
cpack --preset my-windows-release
```

The archive and its SHA-256 checksum land next to the build, in
`build/bin/my-windows-release/`:

```
BayesTS-0.0.1-Windows-AMD64.zip
BayesTS-0.0.1-Windows-AMD64.zip.sha256
```

It contains `bin/bayests.exe`, the runtime libraries it needs, and
`share/doc/BayesTS/` with the licence and this README. Unpack it anywhere and
`bayests` runs without MSYS2, vcpkg or HDF5 installed.

`cmake --build ... --target package` does the same thing if you prefer it to
`cpack`, and `cmake --install ...` writes the same layout into
`CMAKE_INSTALL_PREFIX` without producing an archive.

**Windows installer**

`ZIP` is the only generator configured unconditionally, because it needs
nothing beyond CMake. When [NSIS](https://nsis.sourceforge.io/) is present an
installer is built alongside it:

```
BayesTS-0.0.1-Windows-AMD64.exe
BayesTS-0.0.1-Windows-AMD64.exe.sha256
```

It installs into `%ProgramFiles%\BayesTS`, shows the licence, offers to add
`bayests` to `PATH`, and removes a previous version before installing over it.
Elevation is required, and uninstalling removes the install directory, the
registry entries and the `PATH` entry again.

For an unattended install, `/S` runs the installer without a UI and `/D=` picks
the directory — NSIS requires `/D` last and unquoted, even when the path
contains spaces:

```bat
BayesTS-0.0.1-Windows-AMD64.exe /S /D=C:\tools\BayesTS
"C:\tools\BayesTS\Uninstall.exe" /S
```

A silent install leaves `PATH` alone: the page that asks about it cannot be
answered without a UI, and the script adds nothing unless it is. Put the `bin`
directory on `PATH` yourself if an unattended install needs it.

NSIS does not have to be on `PATH`: configuration also reads the registry key
its installer writes — under both `SOFTWARE\NSIS` and `SOFTWARE\WOW6432Node\NSIS`,
since NSIS is a 32-bit application and a 64-bit CMake reading only the former
finds nothing — and falls back to the default install directories. Point at it
by hand if it is somewhere unusual:

```bash
cmake --preset my-windows-release -DBAYESTS_MAKENSIS_EXECUTABLE="D:/NSIS/makensis.exe"
```

Configuration reports which generators you get, so check for this line:

```
-- NSIS found: D:/NSIS/makensis.exe - 'package' also builds an installer
```

If it says NSIS was not found, only the ZIP is built. A generator whose tool is
missing would otherwise fail the whole `package` target, which is why the list
is assembled at configure time rather than fixed.

**Source package**

```bash
cpack --config build/bin/my-windows-release/CPackSourceConfig.cmake
```

Produces `BayesTS-0.0.1-src.zip` and `.tar.gz`. The ignore list drops the build
tree, `.git/`, `CMakeUserPresets.json` and every `*.h5`, since model files are
derived data and run to hundreds of megabytes.

**Snap package (Linux)**

`snap/snapcraft.yaml` builds `bayests` as a strictly confined snap for amd64 and
arm64. It needs [snapcraft](https://snapcraft.io/snapcraft) and LXD:

```bash
sudo snap install snapcraft --classic
sudo snap install lxd && sudo lxd init --auto

snapcraft                                       # builds for the host architecture
sudo snap install --dangerous ./bayests_0.0.1_amd64.snap
bayests
```

`snapcraft remote-build` builds both architectures on Launchpad instead, which
is the only practical way to produce the arm64 package from an x86 machine. To
publish, register the name once and upload:

```bash
snapcraft login
snapcraft register bayests
snapcraft upload --release=edge ./bayests_0.0.1_amd64.snap
```

The version is not written in `snapcraft.yaml`. It is read out of
`project(VERSION ...)` during the pull step, by the same expression
`release.yml` uses to check a tag, so the snap cannot claim a version the source
does not.

Dependencies come from the base's archive rather than vcpkg — this is the build
the `FindArmadillo` and `FindHDF5` fallbacks exist for. HighFive is cloned from
the tag in `.github/highfive-version`, which is also spelled out as `source-tag`
in `snapcraft.yaml`: **bumping the pin means editing both files.** Nothing joins
them automatically, because reading the version out of the headers cannot
distinguish the 3.0.0 betas from each other.

Confinement is strict, so `bayests` reads and writes below `$HOME` through the
`home` interface and nothing else. Data on another filesystem needs the
`removable-media` interface, which is not connected automatically:

```bash
sudo snap connect bayests:removable-media
```

The `stage-packages` sonames — `libarmadillo12`, `libhdf5-103-1t64` — are those
of `core24`, meaning Ubuntu 24.04, and are what a base bump breaks first. A name
that no longer exists fails at pull with *package not found*; `ldd
prime/usr/bin/bayests` and `dpkg -S` on what it names give the replacements.

**Portable binaries and `-march=native`**

The numeric libraries — `bayests_core` and `algorithms`, everything that links
`bayests_numeric_flags` — are compiled with `-march=native -mtune=native` only
when `BAYESTS_NATIVE_ARCH` is `ON`. Tuning for the machine that happens to be
compiling produces a binary that faults with an illegal instruction on any CPU
without the same instruction set, so the flag is off by default and the
`release` preset keeps it off. The `default` development preset turns it on:
those builds never leave your machine. Do not switch it on for anything you
intend to hand to someone else.

**Bundled runtime libraries**

`BAYESTS_BUNDLE_RUNTIME_DEPS` (on by default on Windows) copies the DLLs the
executable loads — libstdc++, libgomp, libgfortran, OpenBLAS, LAPACK, HDF5 and
the rest — into `bin/`. CMake resolves them by searching the compiler's own
`bin/` and the `bin/` of every `CMAKE_PREFIX_PATH` entry; it does *not* consult
`PATH`. If a library lives somewhere else, the install step fails with an
unresolved dependency, and the fix is to name the directory:

```bash
cmake --preset my-windows-release -DBAYESTS_RUNTIME_DEP_DIRS="D:/somewhere/bin"
```

The generated documentation is not packaged. It is built on demand from the
`docs` target, described next.

## API documentation

The API reference is generated from the doc comments in the source with
[Doxygen](https://www.doxygen.nl/). It is *not* built as part of a normal build,
so Doxygen is not required to compile the project.

```bash
# Configure as usual -- the docs target appears if Doxygen was found
cmake --preset my-windows-default

# Generate the documentation
cmake --build build/bin/my-windows-default --target docs
```

Open `build/bin/my-windows-default/docs/html/index.html` in a browser.
This README serves as the landing page.

During configuration you should see:

```
-- Doxygen 1.x.y found - build the 'docs' target; output in .../docs/html/index.html
```

If Doxygen is missing, configuration still succeeds and prints a note instead;
the `docs` target simply does not exist. To drop the target on purpose,
configure with `-DBAYESTS_BUILD_DOCS=OFF`.

Settings live in [docs/Doxyfile.in](docs/Doxyfile.in), a template that CMake
expands into the build tree. Notable choices:

* `EXTRACT_ALL = YES`, so files that carry no doc comments yet still appear in
  the file and class lists. Worth revisiting once the codebase is documented
  throughout.
* MathJax renders the LaTeX formulas in the doc comments, so no LaTeX
  installation is needed. Viewing the pages does require network access; set
  `USE_MATHJAX = NO` if the documentation has to work offline.
* `WARN_NO_PARAMDOC = YES` reports `@param` names that no longer match the
  signature.
* Class, inheritance and include graphs are drawn as interactive SVG whenever
  CMake finds Graphviz, and silently skipped when it does not. Per-function
  call graphs are off; set `CALL_GRAPH` and `CALLER_GRAPH` to `YES` if you want
  them, at the cost of a much slower run.

### Writing doc comments

Use Javadoc-style blocks on the declaration in the header, or on the definition
where there is only a `.cpp`. `src/core/algorithms/wishart.cpp` is the worked
example. Formulas go between `\f$ ... \f$` inline or `\f[ ... \f]` for display.

```cpp
/**
 * @brief One-line summary, ending with a period.
 *
 * Longer description, including the statistical model where that is what the
 * function implements.
 *
 * @param x What it holds, including its dimensions.
 * @return What comes back.
 */
```

## Multi-threading

BayesTS uses **OpenMP** and **Armadillo** with **OpenBLAS** for multi-threaded
matrix operations, which can significantly speed up computations on multi-core
CPUs. The following operations are parallelised automatically:

- Matrix multiplication (e.g. `z * a`, `arma::trans(z) * u_sigma_inv_diag * z`)
- Linear system solving (`arma::solve()`)
- Cholesky decomposition (`arma::chol()`)
- Eigenvalue decomposition (`arma::eig_sym()`)
- Kronecker products (`arma::kron()`)
- Wishart distribution sampling (`arma::wishrnd()`)

OpenMP support is picked up by CMake automatically — `-fopenmp` on GCC/MinGW,
`/openmp` on MSVC, and `libomp` where Clang needs it installed separately. A
multi-threaded OpenBLAS is assumed, which is what vcpkg provides. During
configuration you should see:

```
-- OpenMP found - enabling multi-threaded matrix operations
-- OpenMP enabled: Using X threads for matrix operations
```

### Controlling the thread count

By default the program uses all available cores, detected at startup and
reported on the console. `OMP_NUM_THREADS` overrides it:

**Windows (PowerShell)**

```powershell
$env:OMP_NUM_THREADS=4
.\bayests.exe posterior model.h5
```

**Windows (Command Prompt)**

```cmd
set OMP_NUM_THREADS=4
bayests.exe posterior model.h5
```

**Linux/Mac**

```bash
export OMP_NUM_THREADS=4
./bayests posterior model.h5
```

The count can also be fixed in the code, in [src/bayests.cpp](src/bayests.cpp):

```cpp
#ifdef _OPENMP
    omp_set_num_threads(4);  // Use 4 threads instead of auto-detection
    std::cout << "OpenMP enabled: Using 4 threads" << std::endl;
#endif
```

### Performance expectations

Speedup depends on the matrix size, the number of physical cores and the
operation — matrix multiplication and linear solves benefit most:

- **Small matrices** (< 100x100): minimal speedup, overhead dominates
- **Medium matrices** (100x100 to 1000x1000): 2-4x on 4-8 cores
- **Large matrices** (> 1000x1000): 4-8x on 8+ cores

Multi-threading can be *slower* with very small matrices, on systems with
limited memory bandwidth, or when hyperthreading gives more logical cores than
physical ones. Reducing `OMP_NUM_THREADS` to the number of physical cores is
the fix.

### Troubleshooting

*"OpenMP not found" during CMake configuration.* On GCC/MinGW, check
`gcc --version` for a recent enough compiler (>= 9.0); under MSYS2 the runtime
is `pacman -S mingw-w64-x86_64-openmp`.

*Single-threaded despite OpenMP being found.* Check that OpenBLAS itself is
multi-threaded (`vcpkg list | grep openblas`), look for the OpenMP message in
the console output at startup, and try setting `OMP_NUM_THREADS` explicitly. To
confirm threading is working, watch CPU usage during a run, or compare timings:

```bash
OMP_NUM_THREADS=1 ./bayests posterior model.h5
OMP_NUM_THREADS=8 ./bayests posterior model.h5
```

### References

- [Armadillo Documentation - OpenMP](https://arma.sourceforge.net/faq.html#openmp)
- [OpenMP API Specification](https://www.openmp.org/)
- [OpenBLAS Performance Guide](https://github.com/xianyi/OpenBLAS/wiki)

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) is the guide: how a model is split across the
four layers and what each one may touch, the order to add the files in so that
every step builds on its own, how to check a change to a sampler against the
recorded fingerprints, and the documentation conventions.

The short version of the layering, because it is the constraint most likely to
be broken by accident: `include/bayests/` holds the contract, `src/core/` the
numerics with no I/O and no console, `src/io/hdf5/` the only code that may
include HighFive, and `src/models/` the front-end the command line drives.
Treat a new dependency across those lines as a design change rather than a build
fix.

The samplers are stochastic, so a refactor is checked by pinning the RNG and
comparing fingerprints before and after — see `test/golden_models.cpp`. Note
that the `*.h5` fixtures are not in the repository; `test/make_model_fixture.cpp`
and `test/make_varsel_fixture.cpp` generate the inputs.

Contributions are accepted under the BSD 3-Clause terms below. New source files
need the `SPDX-License-Identifier: BSD-3-Clause` header that every existing one
carries.

## License

BayesTS is released under the [BSD 3-Clause License](LICENSE). Every source file
carries an `SPDX-License-Identifier: BSD-3-Clause` header.

BSD-3-Clause was chosen over Apache-2.0 for downstream compatibility: BSD-3-Clause
can be combined with GPL-2.0-only code, and Apache-2.0 cannot. That matters for
R-ecosystem consumers, where `GPL-2` and `GPL (>= 2)` package licences are common.
The Apache-2.0 patent grant offers little in return for a numerical library
implementing published algorithms.

### Dependency licenses

| Dependency | License |
| --- | --- |
| Armadillo | Apache-2.0 |
| HDF5 | BSD-3-Clause style (HDF Group) |
| HighFive | Boost Software License 1.0 |
| OpenBLAS / LAPACK | BSD-3-Clause |
| libgomp (OpenMP, GCC/MinGW builds) | GPL-3 **with GCC Runtime Library Exception** |

The GCC Runtime Library Exception means that linking the OpenMP runtime does not
impose the GPL on this project or on anything built with it. The OpenMP row is
toolchain-specific: MSVC links `vcomp` (proprietary, redistributable) and Clang
links `libomp` (Apache-2.0 with LLVM exception). The dependency set is entirely
permissive under all three, and binaries may be redistributed under BSD-3-Clause
terms.

Armadillo's Apache-2.0 terms are themselves incompatible with GPL-2.0-only; a
downstream needing that combination would have to replace it. BayesTS's own
source imposes no such restriction.

### Sparse linear algebra

Sparse storage and arithmetic use Armadillo's `arma::sp_mat`. Sparse *factorisation*
is handled by the LAPACK banded routines already reachable through the linked
BLAS/LAPACK (`dpttrf`/`dpttrs` for tridiagonal precisions, `dpbtrf`/`dpbtrs` for
banded ones), which are BSD-3-Clause and add no dependency. The precision matrices
in these samplers are tridiagonal or block-banded, so a general sparse Cholesky
would pay graph-analysis overhead to rediscover a structure that is known ahead of
time.

SuiteSparse is deliberately *not* used. Its modules are not uniformly licensed —
`AMD`/`COLAMD` are BSD-3-Clause, CHOLMOD's Cholesky core is LGPL-2.1+, and the
supernodal (BLAS-3) path, `UMFPACK` and `SPQR` are GPL-2.0+ — so the fast path is
the copyleft one. Should a model with a genuinely irregular sparsity pattern be
added later, Eigen's `SimplicialLDLT` is the intended fallback; note that Eigen's
sparse Cholesky and its AMD ordering are LGPL-2.1+ rather than MPL-2.0, so that
step would be an explicit, isolated opt-in rather than a project-wide dependency.

## AI assistance

Parts of this project were written with the help of Claude (Anthropic), used as
a coding assistant for drafting implementations, refactoring and documentation.
Generated code is reviewed before it is committed and is held to the same checks
as anything hand-written: it has to respect the layering, and a change to a
sampler has to reproduce the recorded fingerprints in `test/golden_models.cpp`.
Authorship of the project, and responsibility for the numerics, rest with the
maintainer; the licence and the copyright are unaffected. Commits carrying
assisted work are marked with a `Co-Authored-By` trailer, so the history says
which ones they are.

## References

Durbin, J., & Koopman, S. J. (2002). A simple and efficient simulation smoother
for state space time series analysis. *Biometrika*, 89(3), 603-615.

Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility
with leverage: Fast and efficient likelihood inference. *Journal of
Econometrics*, 140(2), 425-449.
