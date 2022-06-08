#include "test_helper.hpp"

template <typename T> struct test_cosh {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::cosh(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::cosh<T>(cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::cosh<T>(cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_cosh>(Q, 4.42, 2.02);

  test_passes &= test_valid_types<test_cosh>(Q, INFINITY, 2.02);
  test_passes &= test_valid_types<test_cosh>(Q, 4.42, INFINITY);
  test_passes &= test_valid_types<test_cosh>(Q, INFINITY, INFINITY);

  test_passes &= test_valid_types<test_cosh>(Q, NAN, 2.02);
  test_passes &= test_valid_types<test_cosh>(Q, 4.42, NAN);
  test_passes &= test_valid_types<test_cosh>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_cosh>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_cosh>(Q, INFINITY, NAN);
  test_passes &= test_valid_types<test_cosh>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_cosh>(Q, INFINITY, NAN);

  if (!test_passes)
    std::cerr << "cosh complex test fails\n";
}
