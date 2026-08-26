// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_KLGS_2010_IO_H
#define BAYESTS_IO_HDF5_VEC_KLGS_2010_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_klgs_2010
{

/// Builds a sampler input from an HDF5 model file.
///
/// Reads `/data/train/x`, the compact regressors, where the other VECs read
/// `/data/train/z`. The two are different shapes -- tt x n_x against
/// (tt k) x (k n_x) -- so a file written for one model is not silently accepted
/// by the other: whichever dataset is missing leaves the input empty and
/// VecKlgs2010Input::validate() names it.
///
/// Optional groups are skipped rather than demanded: a file without regressors
/// or without forecast data yields an input with those parts left empty, which
/// is exactly how the sampler decides what to run.
VecKlgs2010Input read_input(const HighFive::File &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
VecKlgs2010Draws read_coefficients(const HighFive::File &file);

void write_coefficients(HighFive::File &file, const VecKlgs2010Draws &draws);

} // namespace bayests::hdf5_io::vec_klgs_2010

#endif // BAYESTS_IO_HDF5_VEC_KLGS_2010_IO_H
