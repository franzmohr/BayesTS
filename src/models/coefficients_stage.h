// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_MODELS_COEFFICIENTS_STAGE_H
#define BAYESTS_MODELS_COEFFICIENTS_STAGE_H

#include "io/hdf5/model_io_common.h"

#include <iostream>
#include <string>

/// Whether `coefficients` has anything to do for this model, and says so if not.
///
/// `probe` is the dataset the model's skip has always read. A file whose draws
/// are complete -- or that predates the marker and has the probe -- is skipped,
/// as re-running never re-estimates. One a previous run started writing and did
/// not finish is estimated again, with a line saying why: its draws are not
/// something to keep, and the readers of every later stage refuse them.
///
/// `what` is the word the skip message has always used: "simulation" for a
/// sampler, "estimation" for the two discounted models, which run no chain.
inline bool coefficients_needed(const ModelFile &file, const std::string &probe,
                                const char *what = "simulation")
{
    switch (bayests::hdf5_io::coefficients_state(file, probe))
    {
    case bayests::hdf5_io::CoefficientsState::complete:
        std::cout << "Posterior data already exists in file. Skipping " << what << "."
                  << std::endl;
        return false;
    case bayests::hdf5_io::CoefficientsState::interrupted:
        std::cout << "The posterior in this file is from a run that did not finish writing it. "
                     "Estimating again."
                  << std::endl;
        return true;
    case bayests::hdf5_io::CoefficientsState::absent:
        break;
    }
    return true;
}

#endif // BAYESTS_MODELS_COEFFICIENTS_STAGE_H
