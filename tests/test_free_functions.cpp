#include "test_helper.hpp"

#define test_free_functions(test_name, label, func)                            \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    using T = TestType;                                                        \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T> input = GENERATE(                                                 \
        cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},                      \
        cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},          \
        cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},                \
        cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},    \
        cmplx<T>{inf_val<T>, nan_val<T>});                                     \
                                                                               \
    auto std_in = init_std_complex(input);                                     \
    sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};                \
                                                                               \
    T std_out{};                                                               \
    T h_cplx_out;                                                              \
    auto d_cplx_out = sycl::malloc_device<T>(1, Q);                            \
                                                                               \
    std_out = std::func(std_in);                                               \
                                                                               \
    if (is_type_supported<T>(Q) && is_type_supported<T>(Q)) {                  \
      Q.single_task([=]() {                                                    \
         d_cplx_out[0] = sycl::ext::cplx::func<T>(cplx_input);                 \
       }).wait();                                                              \
      Q.copy(d_cplx_out, &h_cplx_out, 1).wait();                               \
                                                                               \
      check_results(h_cplx_out, std_out);                                      \
    }                                                                          \
                                                                               \
    h_cplx_out = sycl::ext::cplx::func<T>(cplx_input);                         \
                                                                               \
    check_results(h_cplx_out, std_out);                                        \
                                                                               \
    sycl::free(d_cplx_out, Q);                                                 \
  }

test_free_functions("Test free function real cplx", "[real]", real);
test_free_functions("Test free function imag cplx", "[imag]", imag);

#undef test_free_functions

#define test_free_functions(test_name, label, func)                            \
  TEMPLATE_TEST_CASE(test_name, label, (std::pair<double, bool>),              \
                     (std::pair<double, char>), (std::pair<double, int>),      \
                     (std::pair<sycl::half, sycl::half>),                      \
                     (std::pair<float, float>), (std::pair<double, double>)) { \
                                                                               \
    using T = typename TestType::first_type;                                   \
    using X = typename TestType::second_type;                                  \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    X input = GENERATE(4.42, 2.02, inf_val<T>, nan_val<T>);                    \
                                                                               \
    auto std_in = init_deci(input);                                            \
                                                                               \
    T std_out{};                                                               \
    T h_cplx_out;                                                              \
    auto d_cplx_out = sycl::malloc_device<T>(1, Q);                            \
                                                                               \
    std_out = std::func(std_in);                                               \
                                                                               \
    if (is_type_supported<T>(Q) && is_type_supported<X>(Q)) {                  \
      Q.single_task([=]() {                                                    \
         d_cplx_out[0] = sycl::ext::cplx::func<X>(input);                      \
       }).wait();                                                              \
      Q.copy(d_cplx_out, &h_cplx_out, 1).wait();                               \
                                                                               \
      check_results(h_cplx_out, std_out);                                      \
    }                                                                          \
                                                                               \
    h_cplx_out = sycl::ext::cplx::func<X>(input);                              \
                                                                               \
    check_results(h_cplx_out, std_out);                                        \
                                                                               \
    sycl::free(d_cplx_out, Q);                                                 \
  }

test_free_functions("Test free function real deci", "[real]", real);
test_free_functions("Test free function imag deci", "[imag]", imag);

#undef test_free_functions
