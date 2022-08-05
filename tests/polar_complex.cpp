#include "test_helper.hpp"

template <typename T> struct test_polar {
  bool operator()(sycl::queue &Q, T init_rho, T init_theta) {
    bool pass = true;

    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std::complex<T> std_out = std::polar(init_rho, init_theta);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::polar<T>(init_rho, init_theta);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::polar<T>(init_rho, init_theta);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

template <typename T, std::size_t NumElements> struct test_polar_marray {
  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_rho,
                  test_marray<T, NumElements> init_theta) {
    bool pass = true;

    sycl::marray<std::complex<T>, NumElements> std_out{};
    auto *cplx_out = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    sycl::marray<T, NumElements> rho = init_rho.get();
    sycl::marray<T, NumElements> theta = init_theta.get();

    // Get std::complex output
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::polar(rho[i], theta[i]);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::polar<T>(rho, theta);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::polar<T>(rho, theta);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

// Note: Output is undefined if rho is negative or Nan, or theta is Inf
int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_polar>(Q, 4.42, 2.02);
  test_passes &= test_valid_types<test_polar>(Q, 1, 3.14);
  test_passes &= test_valid_types<test_polar>(Q, 1, -3.14);

  // marray test
  constexpr size_t m_size = 4;
  test_marray<double, m_size> A = {1, 4.42, 3, 3};
  test_marray<double, m_size> B = {1, 2.02, 3.14, -3.14};
  test_passes &= test_valid_types<test_polar_marray, m_size>(Q, A, B);

  if (!test_passes)
    std::cerr << "acos complex test fails\n";

  return !test_passes;
}
