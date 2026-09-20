# BayesTS for coding agents

Documentation that teaches an AI coding assistant to write correct BayesTS code:
how a model file is laid out, which algorithm does what, the vocabulary the
attributes are spelled in, the order the pipeline runs in, and how to read the
results back.

BayesTS is driven by one HDF5 file and a four-command CLI, so an assistant
with a shell can already run it. What it does not have is the file format. That
format is the entire API, and it fails silently: a wrong field usually estimates
a different model and reports plausible numbers rather than raising an error.
That is what this directory carries.

It is for **using** BayesTS. Changing BayesTS itself is governed by
[`CLAUDE.md`](../CLAUDE.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) at the
repository root.

## What is in it

```
AGENTS.md                           the rules worth having in context always
skills/bayests/SKILL.md             the skill: vocabulary, traps, dimension arithmetic
skills/bayests/references/
    model-file.md                   every attribute, group and dataset, with shapes
    algorithms.md                   the twenty-two algorithms, and what each refuses
    pipeline.md                     the command line, the run order, exit codes
    results.md                      what a run writes, and reading it from Python and R
    recipes.md                      complete examples: a VAR, a VEC, a DFM, and changes to them
llms.txt                            an index of the above, served at the documentation site's root
.claude-plugin/plugin.json          the Claude Code plugin manifest
```

## Installing it

**Claude Code.** The repository is a plugin marketplace with this directory as
its one plugin:

```
/plugin marketplace add franzmohr/BayesTS
/plugin install bayests@bayests
```

The skill then loads whenever the work touches BayesTS, and pulls in the
reference files only as they are needed.

**Other assistants.** Point the assistant at `AGENTS.md`, or copy it into the
project that uses BayesTS. It links to the reference files and stands on its
own without them. `skills/bayests/` is plain Markdown in the `SKILL.md` layout,
which several assistants besides Claude Code read.

**A chat assistant without the repository.** The documentation site serves
`llms.txt` at its root, with these files beside it.

**Matching a release.** The marketplace tracks `main`. For the description of
the file format that matches a particular release, take `agents/` from that
tag. An installed package carries it under `share/doc/BayesTS/agents/`.

## Keeping it honest

Every Python example in `skills/bayests/` is run by the test suite. The test
`agents.recipes` (`test/check_agent_recipes.py`) executes each example, runs
`bayests posterior` over the file it wrote, and checks the shapes the text
states against the ones the run produced. An example that no scenario claims
fails the test, so a new one cannot go unverified. Every file an example writes
must also pass `bayests check` with no warnings, so an example cannot teach a
dataset the model never reads. The test is registered when
CMake finds a Python with h5py and numpy, and CI requires it.

The tables and the prose are not executed. A shape stated in `model-file.md` or
`results.md` is checked only where an example reads that dataset. So when
BayesTS changes, check a claim against a file the binary wrote, not against the
README. Several entries here exist *because* a first draft got them wrong:
`/posterior/psi/coeffs` is `k*k` wide and not `k(k-1)/2`, a factor model's
`/initial/u_sigma_inv` is a flat `(1, k)`, and a constant-coefficient model with
both a covariance block and `varsel` needs selection datasets for the `psi`
block as well.

A change to the model file (a dataset, an attribute, a shape, a refusal) is not
finished until this directory says the same.
