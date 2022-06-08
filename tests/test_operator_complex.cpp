#include "test_helper.hpp"

#define test_op(name, op)                                                      \
  template <typename T> struct name {                                          \
    bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,        \
                    T init_im2) {                                              \
      bool pass = true;                                                        \
                                                                               \
      auto std_in1 = init_std_complex(init_re1, init_im1);                     \
      auto std_in2 = init_std_complex(init_re2, init_im2);                     \
      sycl::ext::cplx::complex<T> cplx_input1{init_re1, init_im1};             \
      sycl::ext::cplx::complex<T> cplx_input2{init_re2, init_im2};             \
                                                                               \
      std::complex<T> std_out{};                                               \
      auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q); \
                                                                               \
      std_out = std_in1 op std_in2;                                            \
                                                                               \
      Q.single_task([=]() {                                                    \
         cplx_out[0] = cplx_input1 op cplx_input2;                             \
       }).wait();                                                              \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);         \
                                                                               \
      cplx_out[0] = cplx_input1 op cplx_input2;                                \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);        \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op(test_add, +);
test_op(test_sub, -);
test_op(test_mul, *);
test_op(test_div, /);

#undef test_op

#define test_op_assign(name, op_assign)                                        \
  template <typename T> struct name {                                          \
    bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,        \
                    T init_im2) {                                              \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init_re1, init_im1);                      \
      sycl::ext::cplx::complex<T> cplx_input{init_re1, init_im1};              \
                                                                               \
      auto std_inout = init_std_complex(init_re2, init_im2);                   \
      auto *cplx_inout =                                                       \
          sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);              \
      cplx_inout[0].real(init_re2);                                            \
      cplx_inout[0].imag(init_im2);                                            \
                                                                               \
      std_inout op_assign std_in;                                              \
                                                                               \
      Q.single_task([=]() { cplx_inout[0] op_assign cplx_input; }).wait();     \
                                                                               \
      pass &= check_results(                                                   \
          cplx_inout[0], std::complex<T>(std_inout.real(), std_inout.imag()),  \
          /*is_device*/ true);                                                 \
                                                                               \
      cplx_inout[0].real(init_re2);                                            \
      cplx_inout[0].imag(init_im2);                                            \
                                                                               \
      cplx_inout[0] op_assign cplx_input;                                      \
                                                                               \
      pass &= check_results(                                                   \
          cplx_inout[0], std::complex<T>(std_inout.real(), std_inout.imag()),  \
          /*is_device*/ false);                                                \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op_assign(test_add_assign, +=);
test_op_assign(test_sub_assign, -=);
test_op_assign(test_mul_assign, *=);
test_op_assign(test_div_assign, /=);

#undef test_op_assign

int main() {
  sycl::queue Q;

  bool test_failed = false;
  
  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_add>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_add>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_add>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &=
        test_valid_types<test_add>(Q, INFINITY, INFINITY, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_add>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_add>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_add>(Q, NAN, NAN, NAN, NAN);

    test_passes &= test_valid_types<test_add>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_add>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &= test_valid_types<test_add>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_add>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Addition operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_sub>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_sub>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_sub>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &=
        test_valid_types<test_sub>(Q, INFINITY, INFINITY, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_sub>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_sub>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_sub>(Q, NAN, NAN, NAN, NAN);

    test_passes &= test_valid_types<test_sub>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_sub>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &= test_valid_types<test_sub>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_sub>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Subtraction operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_mul>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_mul>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_mul>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &=
        test_valid_types<test_mul>(Q, INFINITY, INFINITY, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_mul>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_mul>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_mul>(Q, NAN, NAN, NAN, NAN);

    test_passes &= test_valid_types<test_mul>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_mul>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &= test_valid_types<test_mul>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_mul>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Multiplication operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_div>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_div>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_div>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &=
        test_valid_types<test_div>(Q, INFINITY, INFINITY, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_div>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_div>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_div>(Q, NAN, NAN, NAN, NAN);

    test_passes &= test_valid_types<test_div>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_div>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &= test_valid_types<test_div>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &= test_valid_types<test_div>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Division operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_add_assign>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_add_assign>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_add_assign>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &= test_valid_types<test_add_assign>(Q, INFINITY, INFINITY,
                                                     INFINITY, INFINITY);

    test_passes &= test_valid_types<test_add_assign>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_add_assign>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_add_assign>(Q, NAN, NAN, NAN, NAN);

    test_passes &=
        test_valid_types<test_add_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_add_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &=
        test_valid_types<test_add_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_add_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Addition assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_sub_assign>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_sub_assign>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_sub_assign>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &= test_valid_types<test_sub_assign>(Q, INFINITY, INFINITY,
                                                     INFINITY, INFINITY);

    test_passes &= test_valid_types<test_sub_assign>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_sub_assign>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_sub_assign>(Q, NAN, NAN, NAN, NAN);

    test_passes &=
        test_valid_types<test_sub_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_sub_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &=
        test_valid_types<test_sub_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_sub_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Subtraction assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_mul_assign>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_mul_assign>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_mul_assign>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &= test_valid_types<test_mul_assign>(Q, INFINITY, INFINITY,
                                                     INFINITY, INFINITY);

    test_passes &= test_valid_types<test_mul_assign>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_mul_assign>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_mul_assign>(Q, NAN, NAN, NAN, NAN);

    test_passes &=
        test_valid_types<test_mul_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_mul_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &=
        test_valid_types<test_mul_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_mul_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Multiplication assign operator complex test fails\n";
      test_failed = true;
    }
  }

  {
    bool test_passes = true;
    test_passes &= test_valid_types<test_div_assign>(Q, 4.42, 2.02, -1.5, 3.2);

    test_passes &=
        test_valid_types<test_div_assign>(Q, INFINITY, 2.02, INFINITY, 2.02);
    test_passes &=
        test_valid_types<test_div_assign>(Q, 4.42, INFINITY, 4.42, INFINITY);
    test_passes &= test_valid_types<test_div_assign>(Q, INFINITY, INFINITY,
                                                     INFINITY, INFINITY);

    test_passes &= test_valid_types<test_div_assign>(Q, NAN, 2.02, NAN, 2.02);
    test_passes &= test_valid_types<test_div_assign>(Q, 4.42, NAN, 4.42, NAN);
    test_passes &= test_valid_types<test_div_assign>(Q, NAN, NAN, NAN, NAN);

    test_passes &=
        test_valid_types<test_div_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_div_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    test_passes &=
        test_valid_types<test_div_assign>(Q, NAN, INFINITY, NAN, INFINITY);
    test_passes &=
        test_valid_types<test_div_assign>(Q, INFINITY, NAN, INFINITY, NAN);
    if (!test_passes) {
      std::cerr << "Division assign operator complex test fails\n";
      test_failed = true;
    }
  }

  return test_failed;
}
