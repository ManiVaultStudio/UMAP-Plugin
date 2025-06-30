cmake_minimum_required(VERSION 3.22)

# -----------------------------------------------------------------------------
# Project: UMAP plugin testing
# -----------------------------------------------------------------------------
set(UMAP_TESTS "UmapTests")

# Setup of test build depends on setup of parent project, the plugin itself

PROJECT(${UMAP_TESTS})

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Catch2)

find_package(Catch2 CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(SOURCES
    test_main.cpp
	test_utils.h
)

source_group( UmapTests FILES ${SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_executable(${UMAP_TESTS} ${SOURCES})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

# Include umap 
set(UMAP_PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(${UMAP_TESTS} PRIVATE "${UMAP_PLUGIN_DIR}/src")

# -----------------------------------------------------------------------------
# Target link directories
# -----------------------------------------------------------------------------
target_link_libraries(${UMAP_TESTS} PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${UMAP_TESTS} PRIVATE Eigen3::Eigen)

target_link_libraries(${UMAP_TESTS} PRIVATE knncolle::knncolle_hnsw)

if(${MV_UMAP_USE_OPENMP} AND OpenMP_CXX_FOUND)
	message(STATUS "Link ${UMAP_TESTS} to OpenMP")
	target_link_libraries(${UMAP_TESTS} PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------

# Request C++17
target_compile_features(${UMAP_TESTS} PRIVATE cxx_std_20)
target_compile_features(${UMAP_TESTS} PRIVATE c_std_17)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${UMAP_TESTS} PRIVATE /bigobj)    # for Eigen
endif()

# Instruction sets
mv_check_and_set_AVX(${UMAP_TESTS} ${MV_UMAP_USE_AVX})
