# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

#####################################################################
# API documentation
#
# Adds an optional `docs` target that runs Doxygen over src/ and the
# Markdown files at the repository root. The target is deliberately not
# part of ALL: a normal build must not require Doxygen to be installed.
#
#   cmake --build build/bin/my-windows-default --target docs
#
# Output lands in <binaryDir>/docs/html/index.html, i.e. out of source like
# everything else this project generates.

find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(NOT DOXYGEN_FOUND)
    message(STATUS
        "Doxygen not found - the 'docs' target is unavailable. "
        "Install it from https://www.doxygen.nl/download.html or, on MSYS2, "
        "with 'pacman -S mingw-w64-x86_64-doxygen', then reconfigure.")
    return()
endif()

# Graphviz is optional. Without it Doxygen falls back to text-only class and
# include listings instead of drawing graphs. DOT_PATH is handed to Doxygen
# explicitly so the doc build does not depend on dot being on PATH as well --
# CMake may well have found it somewhere else.
if(TARGET Doxygen::dot)
    set(DOXYGEN_HAVE_DOT YES)
    get_filename_component(DOXYGEN_DOT_PATH "${DOXYGEN_DOT_EXECUTABLE}" DIRECTORY)
    message(STATUS "Graphviz found: ${DOXYGEN_DOT_EXECUTABLE} - graphs enabled")
else()
    set(DOXYGEN_HAVE_DOT NO)
    set(DOXYGEN_DOT_PATH "")
    message(STATUS "Graphviz 'dot' not found - documentation is generated without graphs")
endif()

set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

add_custom_target(docs
    COMMAND Doxygen::doxygen "${DOXYFILE_OUT}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)

message(STATUS
    "Doxygen ${DOXYGEN_VERSION} found - build the 'docs' target; "
    "output in ${DOXYGEN_OUTPUT_DIR}/html/index.html")
