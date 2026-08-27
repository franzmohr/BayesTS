
# Contributing

## Adding new models

A model is split across four places, and which one a given piece of code
belongs in is decided by what it is allowed to touch:

| Where | What goes there |
| --- | --- |
| `include/bayests/<model>.h` | The sampler class, plus its input and result structs in `inputs.h` / `results.h` / `priors.h`. This is the contract a host fills in. |
| `src/core/models/<model>.cpp` | The numerics. Values in, values out: no files, no console, no global state beyond the Armadillo RNG. |
| `src/io/hdf5/<model>_io.h/.cpp` | Translation between the model file on disk and those structs. The only layer that may include HighFive. |
| `src/models/<Model>.cpp` | The `BaseModel` front-end the command line drives: decide what still needs computing, read, call the sampler, write back. Reached through `models.h` and registered in `model_factory.cpp`. |

The layering is the point, so treat a new dependency across it as a design
change rather than a build fix. In particular `bayests_core` deliberately links
neither HDF5 nor HighFive and prints nothing: an embedded host such as an R
package either cannot provide those or is not allowed to let a library use them.
Progress and cancellation reach the sampler through a `Reporter` instead.

Each model's readers live in their own `bayests::hdf5_io::<model>` namespace —
they all declare a `read_input`, and the return type is the only thing that
tells them apart.

## The order to add the files in

Work from the bottom of the layering upwards. Each step below builds and links
on its own, so a mistake surfaces in the layer that caused it; going the other
way means the first thing that compiles is the last thing you wrote, and the
structs underneath get reshaped several times on the way there.

1. **The structs, in `include/bayests/`.** 
   * `inputs.h` gets `<Model>Input` and `<Model>Initial`,
   * `results.h` gets `<Model>Draws`,
   * `priors.h` gets any prior block that is new rather than shared, and
   * `spec.h` only if the model needs a knob no other model has.
   Settle the member names and the matrix dimensions here and write them into
   the comments: everything in the three steps that follow is written against
   them, so a rename at this point costs one file and a rename after step 4 costs five.
2. **The sampler declaration, `include/bayests/<model>.h`.** One class,
   `<Model>Sampler`, with `draw_coefficients`, `forecast` and `log_likelihood`.
   This is the whole contract an embedded host sees; nothing below `src/` is
   public.
3. **The numerics, `src/core/models/<model>.cpp`, plus its line in
   `src/core/CMakeLists.txt`.** First real checkpoint: `bayests_core` links
   neither HDF5 nor HighFive, so if this compiles, the model is host-embeddable,
   and if it does not compile without them in scope, you have found a layering
   problem while it is still cheap to fix.
4. **The HDF5 translation, `src/io/hdf5/<model>_io.h/.cpp`, plus its line in
   `src/io/hdf5/CMakeLists.txt`.** `read_input`, `read_coefficients` and
   `write_coefficients` in `bayests::hdf5_io::<model>`. It exists only to fill
   the structs from step 1 and to flip the layout at the boundary — draws in
   columns inside the sampler, draws in rows on disk — which is why it comes
   after the structs are final rather than being drafted alongside them.
5. **The front-end: `src/models/models.h`, `src/models/<Model>.cpp`, and the
   entry in `src/models/model_factory.cpp`, plus its line in
   `src/models/CMakeLists.txt`.** The factory key is the string stored in the
   model file's `/model/algorithm` attribute, so registering it is the single
   step that makes the model reachable from the command line. Leaving it for
   last means a half-finished sampler cannot be invoked by accident.
6. **The fixture, `test/make_model_fixture.cpp`**, unless the model has recorded
   inputs checked into `test/`. `test/golden_models.cpp` itself needs no change:
   it dispatches through `create_model` on `/model/algorithm`, so step 5 is what
   wires the harness up. Record the fingerprints once the numbers look right —
   from then on they are what a refactor is checked against.

Note that steps 3 to 5 each add a line to a different `CMakeLists.txt`. A
forgotten one shows up as an undefined reference at link time rather than as a
compile error, so it is worth adding the file to its target in the same edit
that creates the file.

Naming follows the layer: snake_case under `src/core/` and `src/io/`,
PascalCase under `src/models/` and for the type names. The model's own name
should appear in full in every one of them — a struct called
`CoefficientDraws` rather than `VarNormalWishartDraws` reads as if it were
shared, and the next model to be added will be written on that assumption.

## Checking a change to the samplers

Run the suite first — it needs nothing but a completed build:

```bash
ctest --test-dir build/bin/<your-preset> --output-on-failure
```

That is a smoke test: every sampler and code path runs, and a fixture that
throws fails the test. It will not catch a change in the *numbers*, which is
what most regressions in this codebase look like.

For that, the samplers being stochastic, the only practical method is to pin the
RNG and compare fingerprints before and after. `bayests_golden` prints one per
posterior dataset, so:

```bash
ctest --test-dir build/bin/<your-preset> -R golden -V > before.txt
# make the change, rebuild
ctest --test-dir build/bin/<your-preset> -R golden -V > after.txt
diff before.txt after.txt
```

Same machine, same build, both times — fingerprints are only comparable within
one toolchain, which is why none are checked in as expected values. A clean diff
means the change did not move any number; any hunk is either the effect you
intended or a regression, and it is worth being able to say which before opening
a pull request.

Two models have no generator, because `test/make_model_fixture.cpp` cannot write
them — `VarNormalWishart` and `VecNormalWishart`, not every VEC: the other six
are generated like the VAR models, from the VEC block at the bottom of that
file, as is `DfmNormalGamma` from the DFM block below it. Their tests are
registered only when pointed at a recorded model file, and are skipped with a
note at configure time otherwise:

```bash
cmake --preset <your-preset> \
      -DBAYESTS_WISHART_FIXTURE=path/to/VarNormalWishart.h5 \
      -DBAYESTS_VEC_FIXTURE=path/to/VecNormalWishart.h5
```

`VarNormalWishart` derives three fixtures from its file, one per selection
scheme, and so has a generation step; `VecNormalWishart` reads its file as it
stands. Neither is at risk of being modified: `bayests_golden` stages its own
copy under the temporary directory and clears the posterior there, so a recorded
model file is only ever read.

There is also a coverage gap worth knowing rather than a configuration one: no
fixture yet has a forecast horizon for `VecNormalWishart`. The path it would
exercise -- rewriting the draws as a level VAR and forecasting from those -- is
covered by the other six VECs, which all reach the same code, so what is
missing is that model's own conversion rather than the shared recursion. Note
that supplying `h` is not enough on its own: `/data/forecast/z` has to be in the
level layout, with `p + 1` blocks of endogenous lags, not the differenced layout
`/data/train/z` uses.

A note on what the golden harness does *not* catch, since it has cost real bugs
twice now. `bayests_golden` exits non-zero only if a fixture throws all the way
out, and the `BaseModel` front-ends catch every exception and print it to
stderr. A sampler that fails on one of the three subcommands therefore leaves
that dataset absent and the test green. When adding a fixture, read the
fingerprints once and check that every dataset the model should carry is there --
`absent` next to a `/posterior/forecast` in a fixture whose `h` is positive is a
failure, not a configuration.

Fixtures are generated into the build tree. Nothing writes a `*.h5` into the
source tree, and nothing should — see `test/CMakeLists.txt` for the generation
matrix and add to it when a new branch needs covering.

## Recording the change

Whatever that diff told you, write it down in `CHANGELOG.md` under *Unreleased*.
The answer to "did this move the numbers" is expensive to obtain — a build of the
old sources on the same machine — and impossible to recover later from the diff
alone, so it is worth the two lines it takes to state.

The rule the file asks for is that every entry touching a sampler says whether
draws are unchanged, change by a rounding error, or change; and if they change,
which models and configurations, and why the new numbers are the right ones. A
commit message is not a substitute. A refactor and a fix that moves a posterior
travel in the same commit more often than anyone intends, and the message
normally describes only the refactor.

This matters more than it would in a standalone library because the core is
vendored — copied, not linked — into downstream packages that keep their own
release notes and have to tell their users the same thing. An entry that omits
the effect on draws just moves the work to whoever propagates the change.

## Including Armadillo

Include `"bayests/arma.h"`, never `<armadillo>` directly. It is the one place
Armadillo enters the project, which is what lets an embedded host redirect all
of it at once by defining `BAYESTS_ARMA_HEADER`:

```cmake
target_compile_definitions(... PRIVATE BAYESTS_ARMA_HEADER=<RcppArmadillo.h>)
```

An R package has to do exactly that, because RcppArmadillo must be the first
thing every translation unit sees for the samplers to draw from R's RNG rather
than Armadillo's. A new `#include <armadillo>` anywhere in `include/` or `src/`
silently breaks that, and it breaks it in the host rather than here, so it will
not show up in this project's own build.

## Documenting code

Public functions and classes carry Javadoc-style Doxygen comments. Put the
block on the declaration in the header where there is one, otherwise on the
definition. State matrix dimensions in the `@param` text, and write the
underlying statistics with `\f$ ... \f$` or `\f[ ... \f]`.
`src/core/algorithms/wishart.cpp` is the reference example.

Check your comments render before opening a pull request:

```bash
cmake --build build/bin/<your-preset> --target docs
```

Doxygen warns about `@param` names that do not match the signature, so a clean
run means the comments still describe the code. Details are in the
[README](README.md#api-documentation).

## Using an AI assistant

Contributions drafted with an AI assistant are welcome, and are reviewed as any
other contribution is: you are the author of the patch and answerable for what
it does. Say so in the pull request and add a trailer to the commits, so the
history records it rather than the pull request alone:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

Two things are worth checking with more care than usual in generated code. The
first is the layering — a plausible-looking patch will reach for `std::cout` or
HighFive inside `src/core/`, which compiles locally and breaks the embedded
host. The second is the numerics: a sampler that runs and produces sensible
numbers can still be wrong, so pin the RNG and compare fingerprints before and
after as described above, and check the statistics against the paper rather
than against the comment the assistant wrote for it.