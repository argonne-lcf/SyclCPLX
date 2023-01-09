add_executable(bench_ops)
target_sources(bench_ops PRIVATE bench_ops.cpp)
target_link_libraries(bench_ops PRIVATE benchmark::benchmark)
target_include_directories(bench_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)

add_executable(bench_functions)
target_sources(bench_functions PRIVATE bench_functions.cpp)
target_link_libraries(bench_functions PRIVATE benchmark::benchmark)
target_include_directories(bench_functions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
