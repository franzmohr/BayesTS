// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_UC_IO_COMMON_H
#define BAYESTS_IO_HDF5_UC_IO_COMMON_H

// The part of an unobserved components model's file every member of the family
// reads and writes the same way: the series, the drift's deterministic terms,
// the three coefficient blocks, the irregular component, and the trend and
// cycle paths. Each model's reader adds its error block; see
// uc_normal_gamma_io.cpp and its siblings.

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::uc_common
{

/// Fills the data and the shared priors and starts of `input`, whichever of the
/// family it is. `algorithm` names the model in a refusal.
template <typename Input>
void read_input(const ModelFile &file, Input &input, const char *algorithm)
{
    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_test_observations(file, input.spec, input.test.y);
    read_constraints(file, "/data/train/constraints", input.train.constraints);
    read_constraints(file, "/data/test/constraints", input.test.constraints);
    refuse_constraints(file, "/data/forecast/constraints", algorithm);

    // The drift's deterministic terms, in the layout a factor model reads its
    // own from: tt x n under /data/train/x and h x n under /data/forecast/x.
    read_factor_deterministic(file, input.train, input.forecast);
    read_normal_block_if_present(file, "c", input.c_prior, input.initial.c);

    // Only draw_coefficients needs the priors and the starting values. A file
    // that holds nothing but a fitted posterior can still be forecast from, so a
    // missing one is left for validate() to complain about if it turns out to
    // matter.
    if (input.spec.p > 0)
    {
        read_normal_block_if_present(file, "a", input.a_prior, input.initial.a);
    }
    read_normal_block_if_present(file, "tau0", input.tau0_prior, input.initial.tau0);

    if (input.spec.irregular)
    {
        if (file.exist("/priors/irregular/shape"))
        {
            input.irregular_prior = read_gamma_prior(file, "/priors/irregular");
        }
        read_vec_if_present(file, "/initial/irregular_inv", input.initial.irregular_inv);
    }
}

/// Reads what UcNormalGamma and UcNormalWishart add to the family: the damped
/// trend's rho and the signal equations -- their data, prior and start -- each
/// only where the spec asks for it.
template <typename Input>
void read_extension(const ModelFile &file, Input &input)
{
    if (input.spec.trend == UcTrend::damped)
    {
        read_normal_block_if_present(file, "rho", input.rho_prior, input.initial.rho);
    }
    if (input.spec.n_signals > 0)
    {
        // tt x n_signals and tt x n_signal_x, in the layout of /data/train/y.
        read_mat_if_present(file, "/data/train/signal", input.train_signal);
        read_mat_if_present(file, "/data/train/signal_x", input.train_signal_x);
        read_normal_block_if_present(file, "signal", input.signal_prior, input.initial.signal);
        if (file.exist("/priors/signal_noise/shape"))
        {
            input.signal_noise_prior = read_gamma_prior(file, "/priors/signal_noise");
        }
        read_vec_if_present(file, "/initial/signal_noise_inv", input.initial.signal_noise_inv);
    }
    else if (file.exist("/data/train/signal"))
    {
        // Read so that validate() can say the file carries signals nothing
        // reads, rather than ignore them.
        read_mat_if_present(file, "/data/train/signal", input.train_signal);
    }
}

/// Reads back and writes the datasets read_extension()'s blocks produce.
template <typename Draws>
void read_extension_draws(const ModelFile &file, Draws &draws)
{
    read_draws_if_present(file, "/posterior/rho/coeffs", draws.rho);
    read_draws_if_present(file, "/posterior/signal/coeffs", draws.signal);
    read_draws_if_present(file, "/posterior/signal_noise_inv/coeffs", draws.signal_noise_inv);
}

template <typename Draws>
void write_extension_draws(const ModelFile &file, const Draws &draws)
{
    if (draws.rho.n_elem > 0)
    {
        write_draws(file, "/posterior/rho/coeffs", draws.rho);
    }
    if (draws.signal.n_elem > 0)
    {
        write_draws(file, "/posterior/signal/coeffs", draws.signal);
        write_draws(file, "/posterior/signal_noise_inv/coeffs", draws.signal_noise_inv);
    }
}

/// Reads back the datasets every member of the family writes.
template <typename Draws>
void read_coefficients(const ModelFile &file, Draws &draws)
{
    read_draws_if_present(file, "/posterior/trend/coeffs", draws.trend);
    read_draws_if_present(file, "/posterior/cycle/coeffs", draws.cycle);
    read_draws_if_present(file, "/posterior/a/coeffs", draws.a);
    read_draws_if_present(file, "/posterior/c/coeffs", draws.c);
    read_draws_if_present(file, "/posterior/tau0/coeffs", draws.tau0);
    read_draws_if_present(file, "/posterior/irregular_inv/coeffs", draws.irregular_inv);
}

/// Writes them. The trend path is part of the posterior rather than a
/// by-product: the forecast starts from its end, and nothing recovers it from
/// the parameters alone. The cycle is written only where an irregular component
/// separates it from y minus the trend.
template <typename Draws>
void write_coefficients(const ModelFile &file, const Draws &draws)
{
    write_draws(file, "/posterior/trend/coeffs", draws.trend);
    if (draws.cycle.n_elem > 0)
    {
        write_draws(file, "/posterior/cycle/coeffs", draws.cycle);
    }
    if (draws.a.n_elem > 0)
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
    }
    if (draws.c.n_elem > 0)
    {
        write_draws(file, "/posterior/c/coeffs", draws.c);
    }
    write_draws(file, "/posterior/tau0/coeffs", draws.tau0);
    if (draws.irregular_inv.n_elem > 0)
    {
        write_draws(file, "/posterior/irregular_inv/coeffs", draws.irregular_inv);
    }
}

} // namespace bayests::hdf5_io::uc_common

#endif // BAYESTS_IO_HDF5_UC_IO_COMMON_H
