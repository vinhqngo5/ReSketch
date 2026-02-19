# src/CMakeLists.txt
add_library(quantile_summary_lib INTERFACE)
target_include_directories(quantile_summary_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Check for LIBXXH_DISPATCH option or environment variable
if(NOT DEFINED LIBXXH_DISPATCH)
    set(LIBXXH_DISPATCH ON)
endif()

# Auto-detect ARM and disable dispatch if not explicitly set
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)" AND NOT DEFINED CACHE{LIBXXH_DISPATCH})
    set(LIBXXH_DISPATCH OFF)
endif()

set(FREQUENCY_SUMMARY_SOURCES
    frequency_summary/geometric_sketch/cpp/src/GeometricSketch.cpp
    frequency_summary/geometric_sketch/cpp/src/DynamicSketch.cpp
    frequency_summary/geometric_sketch/cpp/src/Dictionary.cpp
    frequency_summary/geometric_sketch/cpp/src/countmin.cc
    frequency_summary/geometric_sketch/cpp/src/xxhash.c
    frequency_summary/geometric_sketch/cpp/src/prng.cc
    frequency_summary/geometric_sketch/cpp/src/rand48.cc
    frequency_summary/geometric_sketch/cpp/src/MultiHash.cpp
    frequency_summary/geometric_sketch/cpp/src/GeometricSketchMH.cpp
)

if(LIBXXH_DISPATCH AND NOT LIBXXH_DISPATCH STREQUAL "0")
    list(APPEND FREQUENCY_SUMMARY_SOURCES frequency_summary/geometric_sketch/cpp/src/xxh_x86dispatch.c)
else()
    add_compile_definitions(LIBXXH_DISPATCH=0)
endif()

add_library(frequency_summary_lib STATIC ${FREQUENCY_SUMMARY_SOURCES})
target_include_directories(frequency_summary_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/frequency_summary/geometric_sketch/cpp/include
)

target_link_libraries(frequency_summary_lib INTERFACE quantile_summary_lib)
