# src/CMakeLists.txt
add_library(quantile_summary_lib INTERFACE)
target_include_directories(quantile_summary_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(frequency_summary_lib STATIC
    frequency_summary/geometric_sketch/cpp/src/GeometricSketch.cpp
    frequency_summary/geometric_sketch/cpp/src/DynamicSketch.cpp
    frequency_summary/geometric_sketch/cpp/src/Dictionary.cpp
    frequency_summary/geometric_sketch/cpp/src/countmin.cc
    frequency_summary/geometric_sketch/cpp/src/xxhash.c
    frequency_summary/geometric_sketch/cpp/src/prng.cc
    frequency_summary/geometric_sketch/cpp/src/rand48.cc
    frequency_summary/geometric_sketch/cpp/src/xxhash.c
    frequency_summary/geometric_sketch/cpp/src/xxh_x86dispatch.c
    frequency_summary/geometric_sketch/cpp/src/MultiHash.cpp
    frequency_summary/geometric_sketch/cpp/src/GeometricSketchMH.cpp
)
target_include_directories(frequency_summary_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/frequency_summary/geometric_sketch/cpp/include
)

target_link_libraries(frequency_summary_lib INTERFACE quantile_summary_lib)
