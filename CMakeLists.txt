cmake_minimum_required(VERSION 3.18)
project(CRESSIM_MPM VERSION 2.1.0 LANGUAGES C CXX CUDA)

set(PROJECT_VENDOR "Yafei Ou")
set(PROJECT_DESCRIPTION "CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing")
set(PROJECT_PRODUCT_NAME "CRESSim-MPM")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
    @ONLY
)
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/version.rc.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.rc
    @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(UNIX)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
endif()

enable_language(CUDA)

# All GPUs supported by CUDA Toolkit 12.0 - 12.6
# CUDA arch 100 and 120 require CUDA Toolkit 12.8
set(CUDA_ARCH_LIST 50 52 53 60 61 62 70 72 75 80 86 87 89 90)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} CACHE STRING "List of CUDA architectures to build for")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

option(ENABLE_DEBUG_LOGGER "Enable debug logger" ON)
option(ENABLE_EXAMPLES "Enable examples" ON)
option(ENABLE_TESTS "Enable tests" ON)
option(ENGINE_STATIC "Build the engine as a static library"  ON)

set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path to Eigen3 include directory")
set(GLFW_LIBRARY_DIR "" CACHE PATH "Path to the directory containing GLFW .lib/.dll files on Windows")

set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

if(WIN32)
    message(STATUS "Configuring for Windows")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(ENABLE_DEBUG_LOGGER)
    add_compile_definitions(CR_ENABLE_DEBUG_LOGGER)
endif()

if(ENGINE_STATIC)
    add_compile_definitions(CRMPM_ENGINE_STATIC)
endif()

add_subdirectory(src)
add_subdirectory(c_api)

if(ENABLE_EXAMPLES OR ENABLE_TESTS)
    find_package(Eigen3 QUIET)

    # If not found, allow manual specification
    if(NOT Eigen3_FOUND)
        if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
            message(FATAL_ERROR "Eigen3 not found. Set EIGEN3_INCLUDE_DIR to the Eigen include path.")
        endif()
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()

    if(WIN32 AND NOT EXISTS "${GLFW_LIBRARY_DIR}/glfw3.lib")
        message(FATAL_ERROR "GLFW libraries not found. Set GLFW_LIBRARY_DIR.")
    endif()
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_EXAMPLES OR ENABLE_TESTS)
    add_subdirectory(extern/glad/src)
endif()