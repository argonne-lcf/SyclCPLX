#include "test_helper.hpp"

template <typename T> struct test_asin {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::asin(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_asin>(Q, 0.42, 2.02);

  test_passes &= test_valid_types<test_asin>(Q, INFINITY, 2.02);
  test_passes &= test_valid_types<test_asin>(Q, 4.42, INFINITY);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, INFINITY);

  test_passes &= test_valid_types<test_asin>(Q, NAN, 2.02);
  test_passes &= test_valid_types<test_asin>(Q, 4.42, NAN);
  test_passes &= test_valid_types<test_asin>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_asin>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, NAN);
  test_passes &= test_valid_types<test_asin>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, NAN);

  if (!test_passes)
    std::cerr << "asin complex test fails\n";

  return !test_passes;
}
