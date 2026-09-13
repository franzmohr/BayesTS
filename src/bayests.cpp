// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr



#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <iostream>
#include "subcommands.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenBLAS threading control
//
// These are OpenBLAS's own entry points, not part of any BLAS interface, so the
// build defines BAYESTS_HAVE_OPENBLAS only after checking that they actually
// link -- see the OpenBLAS section of the top-level CMakeLists.txt. Without it
// the BLAS thread count is whatever OPENBLAS_NUM_THREADS or the library's own
// default makes it.
#ifdef BAYESTS_HAVE_OPENBLAS
extern "C" {
    void openblas_set_num_threads(int num_threads);
    int openblas_get_num_threads(void);
#ifdef BAYESTS_HAVE_OPENBLAS_GET_PARALLEL
    int openblas_get_parallel(void);
#endif
}
#endif

int main(int argc, char* argv[]) {

#ifdef _OPENMP
    // Get number of available processors
    int num_threads = omp_get_max_threads();

    // Set OpenMP threads (for Armadillo parallel operations)
    omp_set_num_threads(num_threads);

    std::cout << "OpenMP threads: " << num_threads << std::endl;

#ifdef BAYESTS_HAVE_OPENBLAS
    // Set OpenBLAS threads (for BLAS/LAPACK operations) to the OpenMP count,
    // unless the caller has already said how many it wants. Setting it
    // unconditionally overrode OPENBLAS_NUM_THREADS, the usual way to pin BLAS,
    // so OPENBLAS_NUM_THREADS=1 on its own still ran every core -- and the
    // samplers only reproduce single-threaded.
    const char *openblas_env = std::getenv("OPENBLAS_NUM_THREADS");
    if (openblas_env == nullptr || *openblas_env == '\0')
    {
        openblas_set_num_threads(num_threads);
    }

    std::cout << "OpenBLAS threads: " << openblas_get_num_threads();
#ifdef BAYESTS_HAVE_OPENBLAS_GET_PARALLEL
    // The threading model decides what OPENBLAS_NUM_THREADS does: a pthreads
    // build honours it, and an OpenMP build ignores it and follows
    // OMP_NUM_THREADS alone, whatever this program does.
    switch (openblas_get_parallel())
    {
    case 0: std::cout << " (serial)"; break;
    case 1: std::cout << " (pthreads)"; break;
    case 2: std::cout << " (openmp)"; break;
    default: break;
    }
#endif
    std::cout << std::endl;
#endif
#else
    std::cout << "OpenMP not available: Running single-threaded" << std::endl;
#endif

    // Exit codes: 2 for a command line that cannot be acted on, 1 for a run
    // that started and failed. That is what the subcommands do -- see the argc
    // guard at the top of each of coefficients, forecasts and loglik -- and
    // what the smoke tests in .github/workflows/ci.yml and snap.yml assert,
    // since an exit 2 from a bare invocation is the evidence the binary loaded
    // its libraries and reached main() rather than dying in the loader.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <path_to_file.h5 | directory> [args...]\n";
        std::cerr << "Available commands: posterior, coefficients, forecasts, loglik, check\n";
        std::cerr << "              check reads and validates each model without running it\n";
        std::cerr << "Common flags: --group <path>  the group a model's tree hangs under inside\n";
        std::cerr << "                             its file, e.g. /models/3 (default: the root)\n";
        std::cerr << "              --all-groups   run every model below --group rather than the\n";
        std::cerr << "                             one it names (default: the whole file)\n";
        return 2;
    }

    std::map<std::string, std::function<int(int, char**)>> commands = {
        {"coefficients", coefficients},
        {"forecasts", forecasts},
        {"loglik", loglik},
        {"posterior", posterior},
        {"check", check}
    };

    std::string command = argv[1];

    auto it = commands.find(command);
    if (it != commands.end()) {
        return it->second(argc, argv);
    } else {
        // A name that is not a command is the same class of mistake as no name
        // at all, so it exits the same way.
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Available commands: posterior, coefficients, forecasts, loglik, check\n";
        return 2;
    }
}