file(GLOB_RECURSE PROJECT_HEADER
    ${PROJECT_SOURCE_DIR}/include/core/*.h
    ${PROJECT_SOURCE_DIR}/include/core/*.cuh
    ${PROJECT_SOURCE_DIR}/include/math/*.h
    ${PROJECT_SOURCE_DIR}/include/math/*.cuh
    ${PROJECT_SOURCE_DIR}/include/utils/*.h
    ${PROJECT_SOURCE_DIR}/include/utils/*.cuh
    ${PROJECT_SOURCE_DIR}/include/solver/*.h
    ${PROJECT_SOURCE_DIR}/include/solver/*.cuh
    ${PROJECT_SOURCE_DIR}/include/engine/*.h
    ${PROJECT_SOURCE_DIR}/include/engine/*.cuh
    ${PROJECT_SOURCE_DIR}/include/geometry_utils/*.h
    ${PROJECT_SOURCE_DIR}/include/geometry_utils/*.cuh
)

file(GLOB_RECURSE C_API_HEADER
    ${PROJECT_SOURCE_DIR}/c_api/include/*.h
    ${PROJECT_SOURCE_DIR}/c_api/src/*.h
)

file(GLOB_RECURSE C_API_SRC
    ${PROJECT_SOURCE_DIR}/c_api/src/*.cpp
    ${PROJECT_SOURCE_DIR}/c_api/src/*.cu
)

set(TARGET_NAME "crmpm_c_api")

add_library(${TARGET_NAME} SHARED ${C_API_SRC})
target_compile_definitions(${TARGET_NAME} PRIVATE CRMPM_C_API_EXPORTS)

target_sources(${TARGET_NAME} PRIVATE ${PROJECT_HEADER} ${C_API_HEADER})

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/c_api/include)

target_link_libraries(${TARGET_NAME} PRIVATE crmpm_engine)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(WIN32)
    target_sources(${TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR}/generated/version.rc)
endif()
