#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS'S UTILITIES
////////////////////////////////////////////////////////////////////////////////

#define test_op(name, op)                                                      \
  template <typename T, std::size_t NumElements>                               \
  auto test##_##name(sycl::queue &Q, test_marray<T, NumElements> init_re1,     \
                     test_marray<T, NumElements> init_im1,                     \
                     test_marray<T, NumElements> init_re2,                     \
                     test_marray<T, NumElements> init_im2) {                   \
    auto std_in1 = init_std_complex(init_re1.get(), init_im1.get());           \
    auto std_in2 = init_std_complex(init_re2.get(), init_im2.get());           \
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input1 =       \
        sycl::ext::cplx::make_complex_marray(init_re1.get(), init_im1.get());  \
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2 =       \
        sycl::ext::cplx::make_complex_marray(init_re2.get(), init_im2.get());  \
                                                                               \
    sycl::marray<std::complex<T>, NumElements> std_out{};                      \
    auto *cplx_out = sycl::malloc_shared<                                      \
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);         \
                                                                               \
    /* Get std::complex output */                                              \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      std_out[i] = std_in1[i] op std_in2[i];                                   \
                                                                               \
    /* Check cplx::complex output from device */                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { *cplx_out = cplx_input1 op cplx_input2; }).wait(); \
    }                                                                          \
                                                                               \
    check_results(*cplx_out, std_out);                                         \
                                                                               \
    /* Check cplx::complex output from host */                                 \
    *cplx_out = cplx_input1 op cplx_input2;                                    \
                                                                               \
    check_results(*cplx_out, std_out);                                         \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_op(add, +);
test_op(sub, -);
test_op(mul, /);
test_op(div, *);
#undef test_op

#define test_op_assign(name, op_assign)                                        \
  template <typename T, std::size_t NumElements>                               \
  auto test##_##name(sycl::queue &Q, test_marray<T, NumElements> init_re1,     \
                     test_marray<T, NumElements> init_im1,                     \
                     test_marray<T, NumElements> init_re2,                     \
                     test_marray<T, NumElements> init_im2) {                   \
    auto *cplx_inout = sycl::malloc_shared<                                    \
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);         \
                                                                               \
    auto std_in = init_std_complex(init_re1.get(), init_im1.get());            \
    auto std_inout = init_std_complex(init_re2.get(), init_im2.get());         \
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =        \
        sycl::ext::cplx::make_complex_marray(init_re1.get(), init_im1.get());  \
    *cplx_inout =                                                              \
        sycl::ext::cplx::make_complex_marray(init_re2.get(), init_im2.get());  \
                                                                               \
    /* Get std::complex output */                                              \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      std_inout[i] op_assign std_in[i];                                        \
                                                                               \
    /* Check cplx::complex output from device */                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { *cplx_inout op_assign cplx_input; }).wait();       \
    }                                                                          \
                                                                               \
    check_results(*cplx_inout, convert_marray<T>(std_inout));                  \
                                                                               \
    *cplx_inout =                                                              \
        sycl::ext::cplx::make_complex_marray(init_re2.get(), init_im2.get());  \
                                                                               \
    /* Check cplx::complex output from host */                                 \
    *cplx_inout op_assign cplx_input;                                          \
                                                                               \
    check_results(*cplx_inout, convert_marray<T>(std_inout));                  \
  }

test_op_assign(add_assign, +=);
test_op_assign(sub_assign, -=);
test_op_assign(mul_assign, /=);
test_op_assign(div_assign, *=);
#undef test_op_assign

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

#define TEST(func)                                                             \
  TEMPLATE_TEST_CASE_SIG(                                                      \
      "Test marray complex operator " #func, "[arg]",                          \
      ((typename T, std::size_t NumElements), T, NumElements), (double, 14),   \
      (float, 14), (sycl::half, 14)) {                                         \
    sycl::queue Q;                                                             \
                                                                               \
    auto init_re1 = GENERATE(test_marray<T, NumElements>{                      \
        1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,    \
        nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});          \
    auto init_im1 = GENERATE(test_marray<T, NumElements>{                      \
        1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,  \
        nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});          \
                                                                               \
    auto init_re2 = GENERATE(test_marray<T, NumElements>{                      \
        1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,    \
        nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});          \
    auto init_im2 = GENERATE(test_marray<T, NumElements>{                      \
        1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,  \
        nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});          \
                                                                               \
    test##_##func<T, NumElements>(Q, init_re1, init_im1, init_re2, init_im2);  \
  }

TEST(add)
TEST(sub)
TEST(mul)
TEST(div)

TEST(add_assign)
TEST(sub_assign)
TEST(mul_assign)
TEST(div_assign)
#undef TEST
