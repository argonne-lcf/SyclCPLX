#include "test_helper.hpp"

TEMPLATE_TEST_CASE("Test complex asin", "[asin]", double, float, sycl::half) {
  using T = TestType;
  using std::make_tuple;

  sycl::queue Q;

  cmplx<T> input;
  bool is_error_checking;

  std::tie(input, is_error_checking) = GENERATE(table<cmplx<T>, bool>(
      {make_tuple(cmplx<T>{4.42, 2.02}, false),
       make_tuple(cmplx<T>{inf_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, nan_val<T>}, true)}));

  auto std_in = init_std_complex(input);
  sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};

  std::complex<T> std_out{input.re, input.im};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  if (is_error_checking)
    std_out = std::asin(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
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
  }

  check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);

  // Check cplx::complex output from host
  if (is_error_checking)
    cplx_out[0] = sycl::ext::cplx::asin<T>(cplx_input);
  else
    cplx_out[0] = sycl::ext::cplx::sin<T>(sycl::ext::cplx::asin<T>(cplx_input));

  check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);

  sycl::free(cplx_out, Q);
}