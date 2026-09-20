# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

# Writes the `<hash>  <name>` checksum beside each package CPack has just
# built, with LF line endings.
#
# This replaces CPACK_PACKAGE_CHECKSUM rather than correcting it, and the reason
# is ordering. CPack writes its own checksum through a text-mode stream, so on
# Windows the line ends CRLF and `sha256sum -c` reads the carriage return as
# part of the file name: it reports "No such file or directory" and FAILED,
# which looks like a corrupt download rather than a formatting detail. The
# archive is fine; only the side-car is unusable. Post-processing it is not
# open to us -- CPACK_POST_BUILD_SCRIPTS runs after the *package* is written
# and before its checksum is, so a script that rewrote the file would be
# rewriting one that does not exist yet, and CPack would then write over it.
#
# It matters for the released files rather than only for a local build: the
# Windows packages are built on a Windows runner, so the checksums uploaded to
# a release carry the same endings while the Linux ones beside them are LF.
# Whoever verifies a download on anything but Windows meets it.
#
# file(WRITE) does no text-mode translation, so what is written here is what
# lands. The format matches what CPack produced and what `sha256sum` itself
# writes -- the hash, two spaces, and the base name rather than the path, so
# that verifying works in the directory the files were downloaded into.
#
# **The hash is taken from the staged copy and written to the output
# directory**, and those are two different places: CPACK_PACKAGE_FILES names
# the package under `_CPack_Packages/<system>/<generator>/`, which is where it
# has been built but not where it ends up, and CPack copies only the package
# itself out of there. A checksum written beside the staged file is simply left
# behind. CPACK_PACKAGE_DIRECTORY is the destination, and honours `cpack -B`.
#
# **CPACK_PACKAGE_CHECKSUM is off because of this file.** Drop the script and
# the checksums stop being written at all rather than reverting to CRLF ones.

foreach(_package IN LISTS CPACK_PACKAGE_FILES)
    if(EXISTS "${_package}")
        file(SHA256 "${_package}" _hash)
        get_filename_component(_name "${_package}" NAME)
        set(_checksum "${CPACK_PACKAGE_DIRECTORY}/${_name}.sha256")

        # file(WRITE) opens its stream in text mode, so on Windows it turns the
        # newline below into the CRLF this file exists to avoid -- it is the
        # same translation CPack's own writer does. configure_file() is the one
        # copy that takes NEWLINE_STYLE, so the content goes through a scratch
        # file and comes out with the endings asked for.
        #
        # configure_file() also substitutes @VAR@ and ${VAR} as it copies, which
        # is harmless here and would not be for arbitrary content: a package
        # name is CPACK_PACKAGE_FILE_NAME, built from the project name, the
        # version and the system, and a hash is hex.
        file(WRITE "${_checksum}.in" "${_hash}  ${_name}\n")
        configure_file("${_checksum}.in" "${_checksum}" NEWLINE_STYLE UNIX)
        file(REMOVE "${_checksum}.in")
    endif()
endforeach()
