#include "test_helper.hpp"

template <typename T> struct test_asin {
  bool operator()(sycl::queue &Q, T init_re, T init_im,
                  bool is_error_checking = false) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{init_re, init_im};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    if (is_error_checking)
      std_out = std::asin(std_in);

    // Check cplx::complex output from device
    if (is_error_checking) {
      Q.single_task(
          [=]() { cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input); });
    } else {
      Q.single_task([=]() {
        cplx_out[0] =
            sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));
      });
    }
    Q.wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    if (is_error_checking)
      cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);
    else
      cplx_out[0] =
          sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_asin>(Q, 0.42, 2.02);

  test_passes &= test_valid_types<test_asin>(Q, INFINITY, 2.02, true);
  test_passes &= test_valid_types<test_asin>(Q, 4.42, INFINITY, true);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, INFINITY, true);

  test_passes &= test_valid_types<test_asin>(Q, NAN, 2.02, true);
  test_passes &= test_valid_types<test_asin>(Q, 4.42, NAN, true);
  test_passes &= test_valid_types<test_asin>(Q, NAN, NAN, true);

  test_passes &= test_valid_types<test_asin>(Q, NAN, INFINITY, true);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, NAN, true);
  test_passes &= test_valid_types<test_asin>(Q, NAN, INFINITY, true);
  test_passes &= test_valid_types<test_asin>(Q, INFINITY, NAN, true);

  if (!test_passes)
    std::cerr << "asin complex test fails\n";

  return !test_passes;
}
