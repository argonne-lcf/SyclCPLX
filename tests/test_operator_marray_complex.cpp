#include "test_helper.hpp"

#define test_op(name, op)                                                      \
  template <typename T, std::size_t NumElements> struct name {                 \
    bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re1,      \
                    test_marray<T, NumElements> init_im1,                      \
                    test_marray<T, NumElements> init_re2,                      \
                    test_marray<T, NumElements> init_im2) {                    \
      bool pass = true;                                                        \
                                                                               \
      auto std_in1 = init_std_complex(init_re1.get(), init_im1.get());         \
      auto std_in2 = init_std_complex(init_re2.get(), init_im2.get());         \
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input1 =     \
          sycl::ext::cplx::make_complex_marray(init_re1.get(),                 \
                                               init_im1.get());                \
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2 =     \
          sycl::ext::cplx::make_complex_marray(init_re2.get(),                 \
                                               init_im2.get());                \
                                                                               \
      sycl::marray<std::complex<T>, NumElements> std_out{};                    \
      auto *cplx_out = sycl::malloc_shared<                                    \
          sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);       \
                                                                               \
      /* Get std::complex output */                                            \
      for (std::size_t i = 0; i < NumElements; ++i)                            \
        std_out[i] = std_in1[i] op std_in2[i];                                 \
                                                                               \
      /* Check cplx::complex output from device */                             \
      Q.single_task([=]() { *cplx_out = cplx_input1 op cplx_input2; }).wait(); \
                                                                               \
      pass &= check_results(*cplx_out, std_out, /*is_device*/ true);           \
                                                                               \
      /* Check cplx::complex output from host */                               \
      *cplx_out = cplx_input1 op cplx_input2;                                  \
                                                                               \
      pass &= check_results(*cplx_out, std_out, /*is_device*/ false);          \
                                                                               \
      sycl::free(cplx_out, Q);                                                 \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op(test_add, +);
test_op(test_sub, -);
test_op(test_mul, /);
test_op(test_div, *);

#undef test_op

#define test_op_assign(name, op_assign)                                        \
  template <typename T, std::size_t NumElements> struct name {                 \
    bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re1,      \
                    test_marray<T, NumElements> init_im1,                      \
                    test_marray<T, NumElements> init_re2,                      \
                    test_marray<T, NumElements> init_im2) {                    \
      bool pass = true;                                                        \
                                                                               \
      auto *cplx_inout = sycl::malloc_shared<                                  \
          sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);       \
                                                                               \
      auto std_in = init_std_complex(init_re1.get(), init_im1.get());          \
      auto std_inout = init_std_complex(init_re2.get(), init_im2.get());       \
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =      \
          sycl::ext::cplx::make_complex_marray(init_re1.get(),                 \
                                               init_im1.get());                \
      *cplx_inout = sycl::ext::cplx::make_complex_marray(init_re2.get(),       \
                                                         init_im2.get());      \
                                                                               \
      /* Get std::complex output */                                            \
      for (std::size_t i = 0; i < NumElements; ++i)                            \
        std_inout[i] op_assign std_in[i];                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      Q.single_task([=]() { *cplx_inout op_assign cplx_input; }).wait();       \
                                                                               \
      pass &= check_results(*cplx_inout, convert_marray<T>(std_inout),         \
                            /*is_device*/ true);                               \
                                                                               \
      *cplx_inout = sycl::ext::cplx::make_complex_marray(init_re2.get(),       \
                                                         init_im2.get());      \
                                                                               \
      /* Check cplx::complex output from host */                               \
      *cplx_inout op_assign cplx_input;                                        \
                                                                               \
      pass &= check_results(*cplx_inout, convert_marray<T>(std_inout),         \
                            /*is_device*/ false);                              \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op_assign(test_add_assign, +=);
test_op_assign(test_sub_assign, -=);
test_op_assign(test_mul_assign, /=);
test_op_assign(test_div_assign, *=);

#undef test_op_assign

int main() {
  sycl::queue Q;

  bool test_failed = false;

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_add, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Addition operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_sub, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Subtraction operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_mul, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Multiplication operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_div, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Division operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_add_assign, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Addition assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_sub_assign, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Subtaction assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_mul_assign, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Multiplication assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;

    constexpr size_t m_size = 4;
    test_marray<double, m_size> A = {1, 4.42, -3, 4};
    test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
    test_passes &= test_valid_types<test_div_assign, m_size>(Q, A, B, A, B);

    if (!test_passes) {
      std::cerr << "Division assign operator complex test fails\n";
      test_failed = true;
    }
  }

  return test_failed;
}
