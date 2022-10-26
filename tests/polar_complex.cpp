#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex polar", "[polar]", double, float, sycl::half) {
  using T = TestType;
  using std::make_tuple;

  sycl::queue Q;

  // Test cases
  // Note: Output is undefined if rho is negative or Nan, or theta is Inf
  T rho, theta;
  std::tie(rho, theta) = GENERATE(table<T, T>(
      {make_tuple(4.42, 2.02), make_tuple(1, 3.14), make_tuple(1, -3.14)}));

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::polar(rho, theta);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::polar<T>(rho, theta);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::polar<T>(rho, theta);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex polar", "[polar]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 4), (float, 4), (sycl::half, 4)) {
  sycl::queue Q;

  // Test cases
  auto init_rho = GENERATE(test_marray<T, NumElements>{1.0, 4.42, 3, 3.14});
  auto init_theta =
      GENERATE(test_marray<T, NumElements>{1.0, 2.02, 3.5, -3.14});

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  sycl::marray<T, NumElements> rho = init_rho.get();
  sycl::marray<T, NumElements> theta = init_theta.get();

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::polar(rho[i], theta[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::polar<T>(rho, theta);
     }).wait();
  }

  check_results(*cplx_out, std_out);

  // Check cplx::complex output from host
  *cplx_out = sycl::ext::cplx::polar<T>(rho, theta);

  check_results(*cplx_out, std_out);

  sycl::free(cplx_out, Q);
}
