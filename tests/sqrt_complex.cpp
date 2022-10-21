#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex sqrt", "[sqrt]", double, float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  cmplx<T> input = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  auto std_in = init_std_complex(input);
  sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::sqrt(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::sqrt<T>(cplx_input);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::sqrt<T>(cplx_input);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex sqrt", "[sqrt]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // Test cases
  auto init_re = GENERATE(test_marray<T, NumElements>{
      1.0, 4.42, -3, 4.0, 2.02, inf_val<T>, inf_val<T>, 2.02, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});
  auto init_im = GENERATE(test_marray<T, NumElements>{
      1.0, 2.02, 3.5, -4.0, inf_val<T>, 4.42, nan_val<T>, 4.42, nan_val<T>,
      nan_val<T>, inf_val<T>, nan_val<T>, inf_val<T>, nan_val<T>});

  auto std_in = init_std_complex(init_re.get(), init_im.get());
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =
      sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

  sycl::marray<std::complex<T>, NumElements> std_out{};
  auto *cplx_out = sycl::malloc_shared<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::sqrt(std_in[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::sqrt<T>(cplx_input);
     }).wait();
  }

  check_results(*cplx_out, std_out);

  // Check cplx::complex output from host
  *cplx_out = sycl::ext::cplx::sqrt<T>(cplx_input);

  check_results(*cplx_out, std_out);

  sycl::free(cplx_out, Q);
}
