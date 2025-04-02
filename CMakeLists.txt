cmake_minimum_required(VERSION 3.22)

option(MV_UMAP_USE_OPENMP "Use OpenMP - by default ON" ON)
option(MV_UMAP_USE_AVX "Use AVX if available - by default ON" OFF)

# -----------------------------------------------------------------------------
# UMAP Analysis Plugin
# -----------------------------------------------------------------------------
set(UMAPANALYSIS "UMAPAnalysisPlugin")
PROJECT(${UMAPANALYSIS} 
        VERSION 1.1.1
        DESCRIPTION "A ManiVault plugin that wraps LTLA/umappp"
        LANGUAGES CXX)

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

set(MV_UMAP_CMAKE_MODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${MV_UMAP_CMAKE_MODULES_PATH})

# CMake package manager
include(get_cpm)

# re-applying patches is problematic without CPM_SOURCE_CACHE
# see https://github.com/cpm-cmake/CPM.cmake/issues/577
set(CPM_SOURCE_CACHE ${CMAKE_CURRENT_BINARY_DIR}/.cpm-cache)

# wrapper for cmake fetch utils
include(CMakeFetchUtils)

# cpm managed dependencies
include(CMakeCPMDependencies)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets WebEngineWidgets Concurrent REQUIRED)
find_package(ManiVault COMPONENTS Core PointData CONFIG QUIET)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
endif()

find_package(libscran_umappp CONFIG QUIET)

if(NOT libscran_umappp_FOUND AND NOT TARGET libscran::umappp)

    if(NOT eigen_SOURCE_DIR)
        set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
        set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
        set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)
        fetch_content_url(eigen "https://gitlab.com/libeigen/eigen/-/archive/3866cbfbe8622f41b4f9fa17227aaa7a8de13890/eigen-3866cbfbe8622f41b4f9fa17227aaa7a8de13890.tar.gz") # master as of 26/03/25
    endif()

    # the patches enable use to use the local dependency versions
    fetch_cpm_repo(aarand "https://github.com/LTLA/aarand.git" 4e41649bd3b7f86a59d6cef2ddf02fa615111429) # master as of 04-12-24, v1.0.2+
    fetch_cpm_repo(subpar "https://github.com/LTLA/subpar.git" 561e0314c03476517f9ffe42f7282505a220523a) # master as of 04-12-24, v0.3.1+
    fetch_cpm_repo_patch(kmeans "https://github.com/LTLA/CppKmeans.git" 56fbc963a042c44e3f9684f49b177bd86dd3b681 "kmeans.patch") # master as of 04-12-24, v3.1.1+, depends on aarand and subpar
    fetch_cpm_repo_patch(knncolle "https://github.com/knncolle/knncolle.git" 513dc09b8a681274da83df97c3d5b353a2b61c4b "knncolle.patch") # master as of 04-12-24, v2.3.2+, depends on CppKmeans and subpar
    fetch_cpm_repo_patch(irlba "https://github.com/LTLA/CppIrlba.git" 46281abde9047d28f02f8907a996af191d925a16 "irlba.patch") # master as of 04-12-24, v2.0.0+, depends on eigen, aarand and subpar
    fetch_cpm_repo_patch(umappp "https://github.com/libscran/umappp.git" c48f164b4dbd02f48f2e22bdbacc69f27c4e45ab "umappp.patch") # master as of 04-12-24, v2.0.1+, depends on aarand, subpar, CppIrlba and knnolle
endif()

find_package(knncolle_knncolle_hnsw CONFIG QUIET)

if(NOT knncolle_knncolle_hnsw_FOUND AND NOT TARGET knncolle::knncolle_hnsw)
    fetch_cpm_repo_patch(knncolle_hnsw "https://github.com/knncolle/knncolle_hnsw.git" v0.1.1 "knncolle_hnsw.patch")
endif()

find_package(knncolle_knncolle_annoy CONFIG QUIET)

if(NOT knncolle_knncolle_annoy_FOUND AND NOT TARGET knncolle::knncolle_annoy)
    fetch_cpm_repo_patch(knncolle_annoy "https://github.com/knncolle/knncolle_annoy.git" v0.1.1 "knncolle_annoy.patch")
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
# Define the plugin sources
set(UMAPANALYSIS_SOURCES
    src/UMAPAnalysisPlugin.h
    src/UMAPAnalysisPlugin.cpp
    src/SettingsAction.h
    src/SettingsAction.cpp
    src/KnnSettingsAction.h
    src/KnnSettingsAction.cpp
    src/AdvancedSettingsAction.h
    src/AdvancedSettingsAction.cpp
    src/UMAPAnalysisPlugin.json
)

source_group(Plugin FILES ${UMAPANALYSIS_SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
# Create dynamic library for the plugin
add_library(${UMAPANALYSIS} SHARED ${UMAPANALYSIS_SOURCES})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
# Include ManiVault headers, including system data plugins
target_include_directories(${UMAPANALYSIS} PRIVATE "${ManiVault_INCLUDE_DIR}")

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
# Request C++20
target_compile_features(${UMAPANALYSIS} PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${UMAPANALYSIS} PRIVATE /bigobj)    # for Eigen
endif()

# Instruction sets
mv_check_and_set_AVX(${UMAPANALYSIS} ${MV_UMAP_USE_AVX})

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${UMAPANALYSIS} PRIVATE Qt6::Widgets)
target_link_libraries(${UMAPANALYSIS} PRIVATE Qt6::WebEngineWidgets)
target_link_libraries(${UMAPANALYSIS} PRIVATE Qt6::Concurrent)

target_link_libraries(${UMAPANALYSIS} PRIVATE libscran::umappp)
target_link_libraries(${UMAPANALYSIS} PRIVATE knncolle::knncolle_hnsw)
target_link_libraries(${UMAPANALYSIS} PRIVATE knncolle::knncolle_annoy)

target_link_libraries(${UMAPANALYSIS} PRIVATE ManiVault::Core)
target_link_libraries(${UMAPANALYSIS} PRIVATE ManiVault::PointData)

if(${MV_UMAP_USE_OPENMP} AND OpenMP_CXX_FOUND)
    target_link_libraries(${UMAPANALYSIS} PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
# Install the shared plugin library to the "Plugins" folder in the ManiVault install directory
install(TARGETS ${UMAPANALYSIS}
    RUNTIME DESTINATION Plugins COMPONENT PLUGIN # Windows .dll
    LIBRARY DESTINATION Plugins COMPONENT PLUGIN # Linux/Mac .so
)

add_custom_command(TARGET ${UMAPANALYSIS} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
        --install ${CMAKE_CURRENT_BINARY_DIR}
        --config $<CONFIGURATION>
        --component PLUGIN
        --prefix ${ManiVault_INSTALL_DIR}/$<CONFIGURATION>
)

set_target_properties(${UMAPANALYSIS}
    PROPERTIES
    FOLDER AnalysisPlugins
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Automatically set the debug environment (command + working directory) for MSVC
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_property(TARGET ${PROJECT} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/Debug,$<IF:$<CONFIG:RELWITHDEBINFO>,${ManiVault_INSTALL_DIR}/RelWithDebInfo,${ManiVault_INSTALL_DIR}/Release>>)
    set_property(TARGET ${PROJECT} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/Debug/ManiVault Studio.exe",$<IF:$<CONFIG:RELWITHDEBINFO>,"${ManiVault_INSTALL_DIR}/RelWithDebInfo/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/Release/ManiVault Studio.exe">>)
endif()
