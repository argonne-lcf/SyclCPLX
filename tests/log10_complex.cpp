#include "test_helper.hpp"

template <typename T> struct test_log10 {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    auto std_in = init_std(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::log10(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::log10<T>(cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::log10<T>(cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_log10>(Q, 4.42, 2.02);

  test_passes &= test_valid_types<test_log10>(Q, INFINITY, 2.02);
  test_passes &= test_valid_types<test_log10>(Q, 4.42, INFINITY);
  test_passes &= test_valid_types<test_log10>(Q, INFINITY, INFINITY);

  test_passes &= test_valid_types<test_log10>(Q, NAN, 2.02);
  test_passes &= test_valid_types<test_log10>(Q, 4.42, NAN);
  test_passes &= test_valid_types<test_log10>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_log10>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_log10>(Q, INFINITY, NAN);
  test_passes &= test_valid_types<test_log10>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_log10>(Q, INFINITY, NAN);

  if (!test_passes)
    std::cerr << "log10 complex test fails\n";
}
