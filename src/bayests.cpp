// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr



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
    // Set OpenBLAS threads (for BLAS/LAPACK operations)
    openblas_set_num_threads(num_threads);

    std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << std::endl;
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
        std::cerr << "Usage: " << argv[0] << " <command> <path_to_file.h5> [args...]\n";
        std::cerr << "Available commands: posterior, coefficients, forecasts, loglik\n";
        return 2;
    }

    std::map<std::string, std::function<int(int, char**)>> commands = {
        {"coefficients", coefficients},
        {"forecasts", forecasts},
        {"loglik", loglik},
        {"posterior", posterior}
    };

    std::string command = argv[1];

    auto it = commands.find(command);
    if (it != commands.end()) {
        return it->second(argc, argv);
    } else {
        // A name that is not a command is the same class of mistake as no name
        // at all, so it exits the same way.
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Available commands: posterior, coefficients, forecasts, loglik\n";
        return 2;
    }
}