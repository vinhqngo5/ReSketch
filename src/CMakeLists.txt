# src/CMakeLists.txt


add_library(quantile_summary_lib INTERFACE)

target_include_directories(quantile_summary_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(frequency_summary_lib INTERFACE)

target_include_directories(frequency_summary_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(frequency_summary_lib INTERFACE quantile_summary_lib)