#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex tan", "[tan]", double, float, sycl::half) {
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
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::tan(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::tan<T>(cplx_input);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::tan<T>(cplx_input);

  check_results(h_cplx_out, std_out);

  sycl::free(d_cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex tan", "[tan]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // std::complex test cases
  const auto std_in =
      GENERATE(init_std_complex(sycl::marray<std::complex<T>, NumElements>{
          std::complex<T>{1.0, 1.0},
          std::complex<T>{4.42, 2.02},
          std::complex<T>{-3, 3.5},
          std::complex<T>{4.0, -4.0},
          std::complex<T>{2.02, inf_val<T>},
          std::complex<T>{inf_val<T>, 4.42},
          std::complex<T>{inf_val<T>, nan_val<T>},
          std::complex<T>{2.02, 4.42},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{inf_val<T>, inf_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
          std::complex<T>{inf_val<T>, inf_val<T>},
          std::complex<T>{nan_val<T>, nan_val<T>},
      }));

  // sycl::complex test cases
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input[i] =
        sycl::ext::cplx::complex<T>{std_in[i].real(), std_in[i].imag()};
  }

  sycl::marray<std::complex<T>, NumElements> std_out{};
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = std::tan(std_in[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::tan<T>(cplx_input);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::tan<T>(cplx_input);

  check_results(h_cplx_out, std_out);

  sycl::free(d_cplx_out, Q);
}
