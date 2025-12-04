cmake_minimum_required(VERSION 3.22)

option(MV_UMAP_USE_OPENMP "Use OpenMP - by default ON" ON)
option(MV_UMAP_USE_AVX "Use AVX if available - by default OFF" OFF)
option(MV_UMAP_UNIT_TESTS "Create unit tests - by default OFF" OFF)

# Set DOWNLOAD_EXTRACT_TIMESTAMP option to the time of the extraction, added in 3.24
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# -----------------------------------------------------------------------------
# UMAP Analysis Plugin
# -----------------------------------------------------------------------------
set(UMAPANALYSIS "UMAPAnalysisPlugin")
PROJECT(${UMAPANALYSIS} 
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

if(MV_UMAP_USE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        message(STATUS "UMAPAnalysisPlugin: Found OpenMP")
    endif()
endif()

# check umappp dependencies
include(GetExternalDependencies)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
# Define the plugin sources
set(UMAPANALYSIS_PLUGIN
    src/UMAPAnalysisPlugin.h
    src/UMAPAnalysisPlugin.cpp
)

set(UMAPANALYSIS_SETTINGS
    src/SettingsAction.h
    src/SettingsAction.cpp
    src/KnnSettingsAction.h
    src/KnnSettingsAction.cpp
    src/AdvancedSettingsAction.h
    src/AdvancedSettingsAction.cpp
)

set(UMAPANALYSIS_DISTANCES
    src/util/hnsw_space_corr.h
    src/util/knncolle_matrix_parallel.h
    src/util/knncolle_hnsw_parallel.h
)

set(UMAPANALYSIS_AUX
    PluginInfo.json
)

set(UMAPANALYSIS_SOURCES
    ${UMAPANALYSIS_PLUGIN}
    ${UMAPANALYSIS_SETTINGS}
    ${UMAPANALYSIS_DISTANCES}
)

source_group(Plugin FILES ${UMAPANALYSIS_PLUGIN})
source_group(Settings FILES ${UMAPANALYSIS_SETTINGS})
source_group(Distances FILES ${UMAPANALYSIS_DISTANCES})
source_group(Aux FILES ${UMAPANALYSIS_AUX})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
# Create dynamic library for the plugin
add_library(${UMAPANALYSIS} SHARED ${UMAPANALYSIS_SOURCES} ${UMAPANALYSIS_AUX})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
# Include ManiVault headers, including system data plugins
target_include_directories(${UMAPANALYSIS} PRIVATE "${ManiVault_INCLUDE_DIR}")

# Inlcude own sources for cross-platform include consitency
target_include_directories(${UMAPANALYSIS} PRIVATE "${${CMAKE_CURRENT_SOURCE_DIR}/src}")

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

mv_handle_plugin_config(${UMAPANALYSIS})

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Automatically set the debug environment (command + working directory) for MSVC
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_property(TARGET ${UMAPANALYSIS} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/Debug,$<IF:$<CONFIG:RELWITHDEBINFO>,${ManiVault_INSTALL_DIR}/RelWithDebInfo,${ManiVault_INSTALL_DIR}/Release>>)
    set_property(TARGET ${UMAPANALYSIS} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/Debug/ManiVault Studio.exe",$<IF:$<CONFIG:RELWITHDEBINFO>,"${ManiVault_INSTALL_DIR}/RelWithDebInfo/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/Release/ManiVault Studio.exe">>)
endif()

# -----------------------------------------------------------------------------
# Unit testing
# -----------------------------------------------------------------------------
if(${MV_UMAP_UNIT_TESTS})
	MESSAGE( STATUS "Build UMAP unit tests")
	add_subdirectory("test")
endif()
