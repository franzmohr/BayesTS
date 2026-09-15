# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

# The command line's exit codes, where they are a contract: 2 for a command line
# that cannot be acted on, 1 for a run that started and failed.
#
#   cmake -DBAYESTS=<bayests> -DFIXTURE=<model.h5> -DSCRATCH=<dir> -P check_cli.cmake
#
# Every refusal here used to run with exit code 0. A flag the command did not know
# was a warning -- a misspelled --group ran the model at the root of the file --
# and a second path was never looked at. A directory walk that met an entry it
# could not read ended in std::terminate, with neither exit code.
#
# Only `bayests check` reads the fixture, and a refused command line never opens
# it, so the golden test over the same file is unaffected.

foreach(_required BAYESTS FIXTURE SCRATCH)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "check_cli.cmake needs -D${_required}=...")
    endif()
endforeach()

set(_failures 0)

# Runs bayests with ARGN and compares the exit code with `expected`. The output
# lands in _out and _err for a caller that wants to look at it.
function(expect_exit expected what)
    execute_process(
        COMMAND "${BAYESTS}" ${ARGN}
        RESULT_VARIABLE _code
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr)
    if(_code STREQUAL "${expected}")
        message(STATUS "ok: ${what} (exit ${_code})")
    else()
        message(STATUS "FAIL: ${what}: expected exit ${expected}, got ${_code}\n${_stdout}\n${_stderr}")
        math(EXPR _next "${_failures} + 1")
        set(_failures ${_next} PARENT_SCOPE)
    endif()
    set(_out "${_stdout}" PARENT_SCOPE)
    set(_err "${_stderr}" PARENT_SCOPE)
endfunction()

expect_exit(0 "a well-formed check" check "${FIXTURE}")
expect_exit(2 "a misspelled --group" check "${FIXTURE}" --gruop /models/3)
expect_exit(2 "a misspelled --all-groups" check "${FIXTURE}" --all-group)
expect_exit(2 "a second path" check "${FIXTURE}" "${FIXTURE}")
expect_exit(2 "a step flag on a command that runs one step" coefficients "${FIXTURE}" --no-loglik)
expect_exit(2 "--group followed by another flag" check "${FIXTURE}" --group --all-groups)

expect_exit(2 "a flag where the path goes" check --all-groups "${FIXTURE}")
if(NOT _err MATCHES "is where the path goes")
    message(STATUS "FAIL: a flag before the path was not reported as one\n${_err}")
    math(EXPR _failures "${_failures} + 1")
endif()

# OPENBLAS_NUM_THREADS is the caller's to set. The binary used to overwrite it
# with the OpenMP count, so a BLAS pinned to one thread ran on every core.
#
# Asserted only against a pthreads OpenBLAS, the one kind that honours the
# variable. An OpenMP build -- MSYS2's, which the Windows CI job links --
# ignores it and follows OMP_NUM_THREADS alone, so there it reports 2 whatever
# the binary does, and a serial build always reports 1. Neither can tell a
# binary that respects the variable from one that overwrites it.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "${BAYESTS}"
    OUTPUT_VARIABLE _threads_out
    ERROR_QUIET)
if(_threads_out MATCHES "OpenBLAS threads: ([0-9]+) [(]pthreads[)]")
    if(CMAKE_MATCH_1 STREQUAL "1")
        message(STATUS "ok: OPENBLAS_NUM_THREADS is respected")
    else()
        message(STATUS "FAIL: OPENBLAS_NUM_THREADS=1 ran ${CMAKE_MATCH_1} OpenBLAS threads")
        math(EXPR _failures "${_failures} + 1")
    endif()
elseif(_threads_out MATCHES "OpenBLAS threads: [0-9]+ [(]([a-z]+)[)]")
    message(STATUS "skipped: OpenBLAS reports its threading as '${CMAKE_MATCH_1}', which does not honour OPENBLAS_NUM_THREADS")
else()
    message(STATUS "skipped: the binary does not report an OpenBLAS threading model")
endif()

# With neither variable set the binary runs one thread, not one per logical core.
# Unset rather than inherited, since this test's own environment pins both. Every
# OpenBLAS agrees here: a pthreads build is set to the OpenMP count, an OpenMP
# build follows it and a serial one is 1 regardless.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=OMP_NUM_THREADS --unset=OPENBLAS_NUM_THREADS "${BAYESTS}"
    OUTPUT_VARIABLE _default_out
    ERROR_QUIET)
if(_default_out MATCHES "OpenMP threads: ([0-9]+)")
    if(CMAKE_MATCH_1 STREQUAL "1")
        message(STATUS "ok: OpenMP defaults to one thread")
    else()
        message(STATUS "FAIL: with OMP_NUM_THREADS unset OpenMP ran ${CMAKE_MATCH_1} threads")
        math(EXPR _failures "${_failures} + 1")
    endif()
else()
    message(STATUS "skipped: the binary does not report an OpenMP thread count")
endif()
if(_default_out MATCHES "OpenBLAS threads: ([0-9]+)")
    if(CMAKE_MATCH_1 STREQUAL "1")
        message(STATUS "ok: OpenBLAS defaults to one thread")
    else()
        message(STATUS "FAIL: with both variables unset OpenBLAS ran ${CMAKE_MATCH_1} threads")
        math(EXPR _failures "${_failures} + 1")
    endif()
endif()

# A directory holding the model beside a link to a directory that does not
# exist. The link is skipped with a warning -- nothing behind it was missed --
# and the model is still checked. This is the case that ended in std::terminate.
#
# Made as a junction on Windows, which needs no privilege and is what the
# standard library there cannot see through: it reports the junction as a
# directory, and only opening it fails. A symbolic link made by CMake on Windows
# reports as a regular file and would test nothing. Elsewhere a symbolic link,
# which the library reports as not found.
file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}/walk")
file(COPY "${FIXTURE}" DESTINATION "${SCRATCH}/walk")

if(CMAKE_HOST_WIN32)
    file(TO_NATIVE_PATH "${SCRATCH}/walk/dangling" _native_link)
    file(TO_NATIVE_PATH "${SCRATCH}/no-such-target" _native_target)
    execute_process(
        COMMAND cmd /c mklink /J "${_native_link}" "${_native_target}"
        RESULT_VARIABLE _link
        OUTPUT_QUIET ERROR_QUIET)
else()
    file(CREATE_LINK "${SCRATCH}/no-such-target" "${SCRATCH}/walk/dangling" RESULT _link SYMBOLIC)
endif()

if(_link STREQUAL "0")
    expect_exit(0 "a directory walk past a dangling link" check "${SCRATCH}/walk")
    if(NOT _err MATCHES "skipping" OR NOT _out MATCHES "accepted")
        message(STATUS "FAIL: the walk did not warn about the link and check the model\n${_out}\n${_err}")
        math(EXPR _failures "${_failures} + 1")
    endif()
else()
    message(STATUS "skipped: a symbolic link cannot be created here (${_link})")
endif()

file(REMOVE_RECURSE "${SCRATCH}")

if(_failures GREATER 0)
    message(FATAL_ERROR "${_failures} command line check(s) failed")
endif()
