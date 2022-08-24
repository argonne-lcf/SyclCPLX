file(GLOB test_cases CONFIGURE_DEPENDS "*.cpp")

# set(test_cases 
#     "abs_complex.cpp"
#     "acos_complex.cpp"
#     "acosh_complex.cpp"
#     "arg_complex.cpp"
#     "asin_complex.cpp"
#     "asinh_complex.cpp"
#     "atan_complex.cpp"
#     "atanh_complex.cpp"
#     "cos_complex.cpp"
#     "cosh_complex.cpp"
#     "exp_complex.cpp"
#     "log_complex.cpp"
#     "log10_complex.cpp"
#     "norm_complex.cpp"
#     "polar_complex.cpp"
#     "pow_complex.cpp"
#     "sin_complex.cpp"
#     "sinh_complex.cpp"
#     "sqrt_complex.cpp"
#     "tan_complex.cpp"
#     "tanh_complex.cpp"
#     "test_complex_types.cpp"
#     "test_gencomplex.cpp"
#     "test_operator_complex.cpp"
#     "test_stream_operator.cpp"
# )

foreach(test_file IN LISTS test_cases)
    # if(EXISTS "${test_file}")
    get_filename_component(exe_name "${test_file}" NAME_WE)

    add_executable(${exe_name} ${test_file})
    target_include_directories(${exe_name} PUBLIC ../include/)
    target_link_libraries(${exe_name} PRIVATE
        Catch2::Catch2WithMain
    )
    
    catch_discover_tests(${exe_name})
    # else()
        # message(FATAL_ERROR "No file named ${test_file}")
    # endif()
endforeach()
