#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS'S UTILITIES
////////////////////////////////////////////////////////////////////////////////

#define test_op(name, op)                                                      \
  template <typename T, typename X, std::size_t NumElements>                   \
  auto test##_##name(                                                          \
      sycl::queue &Q,                                                          \
      const sycl::marray<std::complex<X>, NumElements> &std_in1,               \
      const sycl::marray<std::complex<X>, NumElements> &std_in2,               \
      const sycl::marray<sycl::ext::cplx::complex<T>, NumElements>             \
          &cplx_input1,                                                        \
      const sycl::marray<sycl::ext::cplx::complex<T>, NumElements>             \
          &cplx_input2) {                                                      \
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
      check_results(*cplx_out, std_out);                                       \
    }                                                                          \
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
  template <typename T, typename X, std::size_t NumElements>                   \
  auto test##_##name(                                                          \
      sycl::queue &Q,                                                          \
      const sycl::marray<std::complex<X>, NumElements> &std_in,                \
      sycl::marray<std::complex<X>, NumElements> &std_inout,                   \
      const sycl::marray<sycl::ext::cplx::complex<T>, NumElements>             \
          &cplx_input1,                                                        \
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements> &cplx_input2) {   \
                                                                               \
    auto *cplx_inout = sycl::malloc_shared<                                    \
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);         \
    *cplx_inout = cplx_input2;                                                 \
                                                                               \
    /* Get std::complex output */                                              \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      std_inout[i] op_assign std_in[i];                                        \
                                                                               \
    /* Check cplx::complex output from device */                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { *cplx_inout op_assign cplx_input1; }).wait();      \
      check_results(*cplx_inout, convert_marray<T>(std_inout));                \
    }                                                                          \
                                                                               \
    *cplx_inout = cplx_input2;                                                 \
                                                                               \
    /* Check cplx::complex output from host */                                 \
    *cplx_inout op_assign cplx_input1;                                         \
                                                                               \
    check_results(*cplx_inout, convert_marray<T>(std_inout));                  \
                                                                               \
    sycl::free(cplx_inout, Q);                                                 \
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
      "Test marray complex operator " #func, "[" #func "]",                    \
      ((typename T, std::size_t NumElements), T, NumElements), (double, 14),   \
      (float, 14), (sycl::half, 14)) {                                         \
    sycl::queue Q;                                                             \
                                                                               \
    /* sycl::half cases are emulated with float for std::complex class         \
     * (std_in) */                                                             \
    using X = typename std::conditional<std::is_same<T, sycl::half>::value,    \
                                        float, T>::type;                       \
                                                                               \
    /* std::complex test cases */                                              \
    const auto std_in1 =                                                       \
        GENERATE(init_std_complex(sycl::marray<std::complex<T>, NumElements>{  \
            std::complex<T>{1.0, 1.0},                                         \
            std::complex<T>{4.42, 2.02},                                       \
            std::complex<T>{-3, 3.5},                                          \
            std::complex<T>{4.0, -4.0},                                        \
            std::complex<T>{2.02, inf_val<T>},                                 \
            std::complex<T>{inf_val<T>, 4.42},                                 \
            std::complex<T>{inf_val<T>, nan_val<T>},                           \
            std::complex<T>{2.02, 4.42},                                       \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{inf_val<T>, inf_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{inf_val<T>, inf_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
        }));                                                                   \
    sycl::marray<std::complex<X>, NumElements> std_in2;                        \
    for (std::size_t i = 0; i < NumElements; i++) {                            \
      std_in2[i] = std::complex<X>{std_in1[i].real(), std_in1[i].imag()};      \
    }                                                                          \
                                                                               \
    /* sycl::complex test cases */                                             \
    const auto cplx_input1 =                                                   \
        GENERATE(sycl::marray<sycl::ext::cplx::complex<T>, NumElements>{       \
            sycl::ext::cplx::complex<T>{1.0, 1.0},                             \
            sycl::ext::cplx::complex<T>{4.42, 2.02},                           \
            sycl::ext::cplx::complex<T>{-3, 3.5},                              \
            sycl::ext::cplx::complex<T>{4.0, -4.0},                            \
            sycl::ext::cplx::complex<T>{2.02, inf_val<T>},                     \
            sycl::ext::cplx::complex<T>{inf_val<T>, 4.42},                     \
            sycl::ext::cplx::complex<T>{inf_val<T>, nan_val<T>},               \
            sycl::ext::cplx::complex<T>{2.02, 4.42},                           \
            sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},               \
            sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},               \
            sycl::ext::cplx::complex<T>{inf_val<T>, inf_val<T>},               \
            sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},               \
            sycl::ext::cplx::complex<T>{inf_val<T>, inf_val<T>},               \
            sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},               \
        });                                                                    \
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2;        \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      cplx_input2[i] = sycl::ext::cplx::complex<T>{cplx_input1[i].real(),      \
                                                   cplx_input1[i].imag()};     \
    }                                                                          \
                                                                               \
    test##_##func<T, X, NumElements>(Q, std_in1, std_in2, cplx_input1,         \
                                     cplx_input2);                             \
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

#define test_unary_op(test_name, label, op)                                    \
  TEMPLATE_TEST_CASE_SIG(                                                      \
      test_name, label,                                                        \
      ((typename T, std::size_t NumElements), T, NumElements), (double, 14),   \
      (float, 14), (sycl::half, 14)) {                                         \
    sycl::queue Q;                                                             \
                                                                               \
    /* std::complex test cases */                                              \
    const auto std_in =                                                        \
        GENERATE(init_std_complex(sycl::marray<std::complex<T>, NumElements>{  \
            std::complex<T>{1.0, 1.0},                                         \
            std::complex<T>{4.42, 2.02},                                       \
            std::complex<T>{-3, 3.5},                                          \
            std::complex<T>{4.0, -4.0},                                        \
            std::complex<T>{2.02, inf_val<T>},                                 \
            std::complex<T>{inf_val<T>, 4.42},                                 \
            std::complex<T>{inf_val<T>, nan_val<T>},                           \
            std::complex<T>{2.02, 4.42},                                       \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{inf_val<T>, inf_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
            std::complex<T>{inf_val<T>, inf_val<T>},                           \
            std::complex<T>{nan_val<T>, nan_val<T>},                           \
        }));                                                                   \
                                                                               \
    /* sycl::complex test cases */                                             \
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input;         \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      cplx_input[i] =                                                          \
          sycl::ext::cplx::complex<T>{std_in[i].real(), std_in[i].imag()};     \
    }                                                                          \
                                                                               \
    sycl::marray<std::complex<T>, NumElements> std_out{};                      \
    auto *cplx_out = sycl::malloc_shared<                                      \
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);         \
                                                                               \
    /* Get std::complex output */                                              \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      std_out[i] = op std_in[i];                                               \
                                                                               \
    /* Check cplx::complex output from device */                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { *cplx_out = op cplx_input; }).wait();              \
                                                                               \
      check_results(*cplx_out, std_out);                                       \
    }                                                                          \
                                                                               \
    /* Check cplx::complex output from host */                                 \
    *cplx_out = op cplx_input;                                                 \
                                                                               \
    check_results(*cplx_out, std_out);                                         \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_unary_op("Test marray complex addition unary operator", "[add]", +);
test_unary_op("Test marray complex subtraction unary operator", "[sub]", -);

#undef test_unary_op
