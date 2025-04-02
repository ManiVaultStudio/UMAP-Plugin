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

include(FetchContent)

# Find the patch program.
find_program(PATCH_EXECUTABLE patch)
if(CMAKE_HOST_WIN32 AND NOT PATCH_EXECUTABLE)
    # The Windows git executable is distributed with patch.exe. Find the path to the executable, if
    # it exists, then search `../usr/bin` and `../../usr/bin` for patch.exe.
    find_package(Git QUIET)
    if(GIT_EXECUTABLE)
        get_filename_component(extra_search_path ${GIT_EXECUTABLE} DIRECTORY)
        get_filename_component(extra_search_path_1up ${extra_search_path} DIRECTORY)
        get_filename_component(extra_search_path_2up ${extra_search_path_1up} DIRECTORY)
        find_program(
        PATCH_EXECUTABLE patch HINTS "${extra_search_path_1up}/usr/bin"
                                     "${extra_search_path_2up}/usr/bin"
        )
    endif()
endif()
if(NOT PATCH_EXECUTABLE)
    message(FATAL_ERROR "Couldn't find `patch` executable to use with PATCHES keyword.")
endif()

function(check_patch_needed target_dir patch_file result_var)
    execute_process(
        COMMAND patch --dry-run -p1 < ${patch_file}
        WORKING_DIRECTORY ${target_dir}
        RESULT_VARIABLE PATCH_NEEDED
        ERROR_QUIET
    )
    set(${result_var} ${PATCH_NEEDED} PARENT_SCOPE)
endfunction()

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
    # avoid several of eigens cmake config steps
    set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
    set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)

    FetchContent_Declare(
      umappp 
      GIT_REPOSITORY https://github.com/libscran/umappp
      GIT_TAG v2.0.1
    )

    FetchContent_MakeAvailable(umappp)
    
    message(STATUS "getting libscran_umappp")
endif()

find_package(knncolle_knncolle_hnsw CONFIG QUIET)

if(NOT knncolle_knncolle_hnsw_FOUND AND NOT TARGET knncolle::knncolle_hnsw)
    FetchContent_Declare(
      knncolle_hnsw
      GIT_REPOSITORY https://github.com/knncolle/knncolle_hnsw
      GIT_TAG v0.1.1
    )
    
    set(knncolle_hnsw_patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/knncolle_hnsw.patch")
    check_patch_needed(${knncolle_hnsw_SOURCE_DIR} ${knncolle_hnsw_patch} PATCH_NEEDED)

    if(PATCH_NEEDED EQUAL 0)
        set(PATCH_COMMAND "${PATCH_EXECUTABLE}" -p1 < ${knncolle_hnsw_patch})
    else()
        set(PATCH_COMMAND "")
    endif()

    FetchContent_MakeAvailable(knncolle_hnsw)
endif()

find_package(knncolle_knncolle_annoy CONFIG QUIET)

if(NOT knncolle_knncolle_annoy_FOUND AND NOT TARGET knncolle::knncolle_annoy)
    FetchContent_Declare(
      knncolle_annoy
      GIT_REPOSITORY https://github.com/knncolle/knncolle_annoy
      GIT_TAG v0.1.1
    )

    set(knncolle_hnsw_annoy "${CMAKE_CURRENT_SOURCE_DIR}/cmake/knncolle_annoy.patch")
    check_patch_needed(${knncolle_annoy_SOURCE_DIR} ${knncolle_hnsw_annoy} PATCH_NEEDED)

    if(PATCH_NEEDED EQUAL 0)
        set(PATCH_COMMAND "${PATCH_EXECUTABLE}" -p1 < ${knncolle_hnsw_annoy})
    else()
        set(PATCH_COMMAND "")
    endif()

    FetchContent_MakeAvailable(knncolle_annoy)
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
