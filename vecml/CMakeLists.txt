# Configure where to find vecml headers and libcpr.so.1
# You can pass either of these on the command line:
#   -DVECML_LIB_PATH=/path/to/vecml
#   -Dvecml-lib-path=/path/to/vecml   (alias; will be mapped to VECML_LIB_PATH)

# Map alias variable (with dash) to canonical cache var
if(NOT DEFINED VECML_LIB_PATH AND DEFINED vecml-lib-path)
	set(VECML_LIB_PATH "${vecml-lib-path}")
endif()

set(VECML_LIB_PATH "${VECML_LIB_PATH}" CACHE PATH "Root path containing vecml headers and libcpr library (include and/or lib under here)")

# Info: show configured root path
if(VECML_LIB_PATH)
	message(STATUS "VECML_LIB_PATH set to: ${VECML_LIB_PATH}")
else()
	message(STATUS "VECML_LIB_PATH not provided; relying on system/default include and library paths")
endif()

add_executable(vecml-bench sdk_utility_test.cc)

# Option: if the external vecml library was built with the old libstdc++ ABI
# (i.e. _GLIBCXX_USE_CXX11_ABI=0) then the symbol mangling won't match the
# default build. If you see undefined symbols referencing
# std::__cxx11::basic_string but the vecml .so exports signatures without
# the __cxx11 inline namespace, set this to ON to force the same ABI.
option(VECML_USE_OLD_CXX_ABI "Build vecml-bench with the old libstdc++ ABI (set to ON if vecml .so uses non-__cxx11 std::string)" OFF)

# Try to locate headers and library. Prefer explicit VECML_LIB_PATH when provided.
set(_VECML_INCLUDE_HINTS)
set(_VECML_LIBRARY_HINTS)
if(VECML_LIB_PATH)
	# In this layout, headers and shared libs live directly under VECML_LIB_PATH
	list(APPEND _VECML_INCLUDE_HINTS "${VECML_LIB_PATH}")
	list(APPEND _VECML_LIBRARY_HINTS "${VECML_LIB_PATH}")
endif()

# Info: show search hints
if(_VECML_INCLUDE_HINTS)
	message(STATUS "vecml include search hints: ${_VECML_INCLUDE_HINTS}")
endif()
if(_VECML_LIBRARY_HINTS)
	message(STATUS "vecml library search hints: ${_VECML_LIBRARY_HINTS}")
endif()

# Find headers (try common names; if unknown, using the path is still fine)
find_path(VECML_INCLUDE_DIR
	NAMES fluffy_interface.h
	PATHS ${_VECML_INCLUDE_HINTS}
	NO_CACHE
)
if(VECML_INCLUDE_DIR)
	target_include_directories(vecml-bench PRIVATE "${VECML_INCLUDE_DIR}")
	message(STATUS "Using vecml include directory: ${VECML_INCLUDE_DIR}")
else()
	# If a root path was provided but we couldn't verify a header, still add it for user includes
	if(VECML_LIB_PATH)
		target_include_directories(vecml-bench PRIVATE "${VECML_LIB_PATH}")
	endif()
	message(STATUS "vecml headers not auto-detected. If compile fails, set -DVECML_LIB_PATH=/path/to/vecml (containing include/) or adjust include dirs.")
endif()

# Find libcpr (shared) — accept both logical name and exact soname
find_library(CPR_LIBRARY
	NAMES cpr libcpr.so.1 libcpr
	PATHS ${_VECML_LIBRARY_HINTS}
)

if(CPR_LIBRARY)
	target_link_libraries(vecml-bench PRIVATE "${CPR_LIBRARY}")
	message(STATUS "Found libcpr: ${CPR_LIBRARY}")
	# Help runtime find the shared library without needing LD_LIBRARY_PATH
	if(VECML_LIB_PATH)
		set_target_properties(vecml-bench PROPERTIES
			BUILD_RPATH "${VECML_LIB_PATH}"
			INSTALL_RPATH "${VECML_LIB_PATH}"
		)
		message(STATUS "Set RPATH for vecml-bench to: ${VECML_LIB_PATH}")
	endif()
else()
	message(FATAL_ERROR "libcpr not found on default paths. If you have it under a custom directory, pass -DVECML_LIB_PATH=/path/to/vecml (libcpr.so.1 under lib/ or the root).")
endif()

# Find and link vecml's main shared library providing FluffyInterface
find_library(FLUFFY_LIBRARY
	NAMES fluffy_shared_lib libfluffy_shared_lib fluffy
	PATHS ${_VECML_LIBRARY_HINTS}
)

if(FLUFFY_LIBRARY)
	target_link_libraries(vecml-bench PRIVATE "${FLUFFY_LIBRARY}")
	message(STATUS "Found fluffy shared library: ${FLUFFY_LIBRARY}")
else()
	message(FATAL_ERROR "fluffy shared library not found in ${_VECML_LIBRARY_HINTS}. Ensure libfluffy_shared_lib.so is present in VECML_LIB_PATH.")
endif()

# If user requested the old C++ ABI, add the define so std::string mangling
# matches the vecml shared library. Use this if you see undefined symbol errors
# referencing std::__cxx11::basic_string while the provided .so exports the
# non-__cxx11 variants.
if(VECML_USE_OLD_CXX_ABI)
    message(STATUS "Building vecml-bench with _GLIBCXX_USE_CXX11_ABI=0 to match vecml .so")
    target_compile_definitions(vecml-bench PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(vecml-bench PRIVATE OpenMP::OpenMP_CXX)
