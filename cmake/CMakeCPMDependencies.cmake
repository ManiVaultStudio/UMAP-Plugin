# fetches dependencies content using CPM (https://github.com/cpm-cmake/CPM.cmake)
include(FetchContent)

macro(fetch_cpm_repo_download REPO_NAME REPO_LINK REPO_TAG)

    CPMAddPackage(
     NAME              ${REPO_NAME}
     GIT_REPOSITORY    ${REPO_LINK}
     GIT_TAG           ${REPO_TAG}
     DOWNLOAD_ONLY     YES
     EXCLUDE_FROM_ALL  YES
    )

    set(${REPO_NAME}_SOURCE_DIR "${${REPO_NAME}_SOURCE_DIR}" CACHE STRING "Dependency module source path")
    message(STATUS "${REPO_NAME}_SOURCE_DIR: ${${REPO_NAME}_SOURCE_DIR}")

endmacro()

macro(fetch_cpm_repo REPO_NAME REPO_LINK REPO_TAG)

    CPMAddPackage(
     NAME              ${REPO_NAME}
     GIT_REPOSITORY    ${REPO_LINK}
     GIT_TAG           ${REPO_TAG}
     EXCLUDE_FROM_ALL  YES
    )

    set(${REPO_NAME}_SOURCE_DIR "${${REPO_NAME}_SOURCE_DIR}" CACHE STRING "Dependency module source path")
    message(STATUS "${REPO_NAME}_SOURCE_DIR: ${${REPO_NAME}_SOURCE_DIR}")

endmacro()

macro(fetch_cpm_repo_patch REPO_NAME REPO_LINK REPO_TAG REPO_PATCH)

    CPMAddPackage(
     NAME              ${REPO_NAME}
     GIT_REPOSITORY    ${REPO_LINK}
     GIT_TAG           ${REPO_TAG}
     EXCLUDE_FROM_ALL  YES
     PATCHES            "${MV_UMAP_CMAKE_MODULES_PATH}/${REPO_PATCH}"
    )

    set(${REPO_NAME}_SOURCE_DIR "${${REPO_NAME}_SOURCE_DIR}" CACHE STRING "Dependency module source path")
    message(STATUS "${REPO_NAME}_SOURCE_DIR: ${${REPO_NAME}_SOURCE_DIR}")

endmacro()
