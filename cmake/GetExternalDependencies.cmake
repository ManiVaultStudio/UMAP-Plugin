find_package(libscran_umappp CONFIG QUIET)

if(NOT libscran_umappp_FOUND AND NOT TARGET libscran::umappp)

    if(NOT eigen_SOURCE_DIR AND NOT TARGET Eigen3::Eigen AND NOT Eigen3_FOUND)
        set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
        set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
        set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)
        fetch_content_url(eigen "https://gitlab.com/libeigen/eigen/-/archive/5.0.1.tar.gz")
    else()
        message(STATUS "UMAPAnalysisPlugin: Using external Eigen")
    endif()

    # the patches enable use to use the local dependency versions
    fetch_content_url(aarand "https://github.com/LTLA/aarand/archive/refs/tags/v1.1.0.zip")
    fetch_content_url(subpar "https://github.com/LTLA/subpar/archive/refs/tags/v0.4.1.zip")
    fetch_content_url(sanisizer "https://github.com/LTLA/sanisizer/archive/refs/tags/v0.1.3.zip")
    fetch_cpm_repo_patch(knncolle "https://github.com/knncolle/knncolle.git" v3.0.1 "knncolle.patch") # depends on subpar
    fetch_cpm_repo_patch(irlba "https://github.com/LTLA/CppIrlba.git" v2.0.2 "irlba.patch") # depends on eigen, aarand and subpar
    fetch_cpm_repo_patch(umappp "https://github.com/libscran/umappp.git" v3.1.0 "umappp.patch") # depends on aarand, subpar, CppIrlba and knnolle
else()
    message(STATUS "UMAPAnalysisPlugin: Using external Umappp")
endif()

find_package(knncolle_knncolle_hnsw CONFIG QUIET)
if(NOT knncolle_knncolle_hnsw_FOUND AND NOT TARGET knncolle::knncolle_hnsw)
    set(KNNCOLLE_HNSW_FETCH_EXTERN OFF)
    find_package(hnswlib CONFIG QUIET)
    if(NOT hnswlib_FOUND AND NOT TARGET hnswlib::hnswlib)
        FetchContent_Declare(
          hnswlib 
          GIT_REPOSITORY https://github.com/nmslib/hnswlib
          GIT_TAG v0.8.0
        )
        FetchContent_MakeAvailable(hnswlib)
    else()
        message(STATUS "UMAPAnalysisPlugin: Using external hnswlib")
    endif()
    fetch_cpm_repo_patch(knncolle_hnsw "https://github.com/knncolle/knncolle_hnsw.git" v0.2.1 "knncolle_hnsw.patch")
endif()

find_package(knncolle_knncolle_annoy CONFIG QUIET)
if(NOT knncolle_knncolle_annoy_FOUND AND NOT TARGET knncolle::knncolle_annoy)
    set(KNNCOLLE_ANNOY_FETCH_EXTERN OFF)
    find_package(Annoy CONFIG QUIET)
    if(NOT Annoy_FOUND AND NOT TARGET Annoy::Annoy)
        FetchContent_Declare(
          Annoy 
          GIT_REPOSITORY https://github.com/spotify/Annoy
          GIT_TAG v1.17.3
        )
        FetchContent_MakeAvailable(Annoy)
    else()
        message(STATUS "UMAPAnalysisPlugin: Using external Annoy")
    endif()
    fetch_cpm_repo_patch(knncolle_annoy "https://github.com/knncolle/knncolle_annoy.git" v0.2.0 "knncolle_annoy.patch")
endif()