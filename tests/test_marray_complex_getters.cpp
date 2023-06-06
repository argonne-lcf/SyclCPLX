#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex real component marray", "[getter]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 8), (float, 8), (sycl::half, 8)) {
  sycl::queue Q;

  // Test cases
  const auto init = GENERATE(sycl::marray<T, NumElements>{
      0.0,
      1.0,
      4.42,
      -0.0,
      -1.0,
      -4.42,
      nan_val<T>,
      inf_val<T>,
  });
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> input;
  for (std::size_t i = 0; i < NumElements; ++i) {
    input[i] = sycl::ext::cplx::complex<T>{init[i], (T)0};
  }

  sycl::marray<T, NumElements> h_out;
  auto d_out = sycl::malloc_device<sycl::marray<T, NumElements>>(1, Q);

  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() { d_out[0] = input.real(); }).wait();
    Q.copy(d_out, &h_out, 1).wait();

    check_results(h_out, init);
  }

  h_out = input.real();

  check_results(h_out, init);

  sycl::free(d_out, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex imag component marray", "[getter]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 8), (float, 8), (sycl::half, 8)) {
  sycl::queue Q;

  // Test cases
  const auto init = GENERATE(sycl::marray<T, NumElements>{
      0.0,
      1.0,
      4.42,
      -0.0,
      -1.0,
      -4.42,
      nan_val<T>,
      inf_val<T>,
  });
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> input;
  for (std::size_t i = 0; i < NumElements; ++i) {
    input[i] = sycl::ext::cplx::complex<T>{(T)0, init[i]};
  }

  sycl::marray<T, NumElements> h_out;
  auto d_out = sycl::malloc_device<sycl::marray<T, NumElements>>(1, Q);

  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() { d_out[0] = input.imag(); }).wait();
    Q.copy(d_out, &h_out, 1).wait();

    check_results(h_out, init);
  }

  h_out = input.imag();

  check_results(h_out, init);

  sycl::free(d_out, Q);
}
