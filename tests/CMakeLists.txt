file(GLOB test_cases CONFIGURE_DEPENDS "*.cpp")

foreach(test_file IN LISTS test_cases)
    if(EXISTS "${test_file}")
        get_filename_component(exe_name "${test_file}" NAME_WE)

        add_executable(${exe_name} ${test_file})
        target_include_directories(${exe_name} PUBLIC ../include/)
        target_link_libraries(${exe_name} PRIVATE
            Catch2::Catch2WithMain
        )

        catch_discover_tests(${exe_name})
    else()
        message(FATAL_ERROR "No file named ${test_file}")
    endif()
endforeach()
