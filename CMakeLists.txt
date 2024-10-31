cmake_minimum_required(VERSION 3.21)

option(MV_UMAP_USE_OPENMP "Use OpenMP - by default ON" ON)
option(MV_UMAP_USE_AVX "Use AVX if available - by default ON" OFF)

# -----------------------------------------------------------------------------
# UMAP Analysis Plugin
# -----------------------------------------------------------------------------
set(UMAPANALYSIS "UMAPAnalysisPlugin")
PROJECT(${UMAPANALYSIS} 
        VERSION 1.1.0
        DESCRIPTION "A ManiVault plugin that wraps LTLA/umappp"
        LANGUAGES CXX)

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

include(FetchContent)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets WebEngineWidgets Concurrent REQUIRED)
find_package(OpenMP)

find_package(ManiVault COMPONENTS Core PointData CONFIG)

# avoid several of eigens cmake config steps
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for Eigen" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)

FetchContent_Declare(
  umappp 
  GIT_REPOSITORY https://github.com/libscran/umappp
  GIT_TAG v2.0.1
)

FetchContent_MakeAvailable(umappp)

FetchContent_Declare(
  knncolle_hnsw
  GIT_REPOSITORY https://github.com/knncolle/knncolle_hnsw
  GIT_TAG v0.1.0
)

FetchContent_MakeAvailable(knncolle_hnsw)

FetchContent_Declare(
  knncolle_annoy
  GIT_REPOSITORY https://github.com/knncolle/knncolle_annoy
  GIT_TAG v0.1.0
)

FetchContent_MakeAvailable(knncolle_annoy)

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
# Request C++17
target_compile_features(${UMAPANALYSIS} PRIVATE cxx_std_17)

if(MSVC)
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

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Automatically set the debug environment (command + working directory) for MSVC
if(MSVC)
    set_property(TARGET ${UMAPANALYSIS} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/debug,${ManiVault_INSTALL_DIR}/release>)
    set_property(TARGET ${UMAPANALYSIS} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/debug/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/release/ManiVault Studio.exe">)
endif()
