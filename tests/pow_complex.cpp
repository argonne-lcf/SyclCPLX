#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Workaround alleged bug in oneAPI where std::pow returns the wrong result
// for certain inf/nan exponents with a finite complex base.

// default implementation, just call real std::pow
template <typename T1, typename T2> auto std_pow(T1 &x, T2 &y) {
  return std::pow(x, y);
}

// speciliazation for complex exponent and base to apply workaround, when
// enabled with SYCL_CLX_TEST_POW_WORKAROUND macro
template <typename T1, typename T2>
auto std_pow(std::complex<T1> &x, std::complex<T2> &y) {
#ifdef SYCL_CPLX_TEST_POW_WORKAROUND
  using ReturnType = decltype(std::pow(x, y));
  return std::exp(ReturnType(y) * ReturnType(std::log(x)));
#else
  return std::pow(x, y);
#endif
}

} // namespace detail

TEMPLATE_TEST_CASE("Test complex pow cplx-cplx overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  cmplx<T> input2 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  auto std_in1 = init_std_complex(input1);
  auto std_in2 = init_std_complex(input2);
  using std_inT = decltype(std_in1);
  sycl::ext::cplx::complex<T> cplx_input1{input1.re, input1.im};
  sycl::ext::cplx::complex<T> cplx_input2{input2.re, input2.im};

  std::complex<T> std_out{};
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  std_out = detail::std_pow(std_in1, std_in2);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);

  sycl::free(d_cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex pow cplx<T>-cplx<U> overload", "[pow]",
                   (std::pair<double, float>), (std::pair<float, sycl::half>),
                   (std::pair<sycl::half, double>)) {
  using T = typename TestType::first_type;
  using X = typename TestType::second_type;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE
  // list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  cmplx<X> input2 = GENERATE(
      cmplx<X>{4.42, 2.02}, cmplx<X>{inf_val<X>, 2.02},
      cmplx<X>{4.42, inf_val<X>}, cmplx<X>{inf_val<X>, inf_val<X>},
      cmplx<X>{nan_val<X>, 2.02}, cmplx<X>{4.42, nan_val<X>},
      cmplx<X>{nan_val<X>, nan_val<X>}, cmplx<X>{nan_val<X>, inf_val<X>},
      cmplx<X>{inf_val<X>, nan_val<X>});

  auto std_in1 = init_std_complex(input1);
  auto std_in2 = init_std_complex(input2);
  sycl::ext::cplx::complex<T> cplx_input1{input1.re, input1.im};
  sycl::ext::cplx::complex<X> cplx_input2{input2.re, input2.im};

  std::complex<T> std_out{};
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  using std_in1T = decltype(std_in1);
  using std_in2T = decltype(std_in2);

  // Get std::complex output
  std_out = detail::std_pow(std_in1, std_in2);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q) && is_type_supported<X>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);

  sycl::free(d_cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex pow cplx-deci overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE
  // list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});
  T input2 = GENERATE(4.42, inf_val<T>, nan_val<T>);

  auto std_in = init_std_complex(input1);
  auto std_deci_in = init_deci(input2);
  sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};
  T deci_input = input2;

  std::complex<T> std_out{};
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = detail::std_pow(std_in, std_deci_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input, deci_input);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input, deci_input);

  CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);

  sycl::free(d_cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex pow deci-cplx overload", "[pow]", double,
                   float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  // Values are generated as cross product of input1 and input2's GENERATE
  // list
  cmplx<T> input1 = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});
  T input2 = GENERATE(4.42, inf_val<T>, nan_val<T>);

  auto std_in = init_std_complex(input1);
  auto std_deci_in = init_deci(input2);
  sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};
  T deci_input = input2;

  std::complex<T> std_out{};
  sycl::ext::cplx::complex<T> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = detail::std_pow(std_deci_in, std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(deci_input, cplx_input);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(deci_input, cplx_input);

  CHECK_COMPLEX_WITHIN_ULP(h_cplx_out, std_out, 1);

  sycl::free(d_cplx_out, Q);
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE_SIG("Test marray complex pow cplx-cplx overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  /* sycl::half cases are emulated with float for std::complex class (std_in)
   */
  using X = typename std::conditional<std::is_same<T, sycl::half>::value, float,
                                      T>::type;

  // std::complex test cases
  const auto std_in1 =
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
  sycl::marray<std::complex<X>, NumElements> std_in2;
  for (std::size_t i = 0; i < NumElements; i++) {
    std_in2[i] = std::complex<X>{std_in1[i].real(), std_in1[i].imag()};
  }

  // sycl::complex test cases
  const auto cplx_input1 =
      GENERATE(sycl::marray<sycl::ext::cplx::complex<T>, NumElements>{
          sycl::ext::cplx::complex<T>{1.0, 1.0},
          sycl::ext::cplx::complex<T>{4.42, 2.02},
          sycl::ext::cplx::complex<T>{-3, 3.5},
          sycl::ext::cplx::complex<T>{4.0, -4.0},
          sycl::ext::cplx::complex<T>{2.02, inf_val<T>},
          sycl::ext::cplx::complex<T>{inf_val<T>, 4.42},
          sycl::ext::cplx::complex<T>{inf_val<T>, nan_val<T>},
          sycl::ext::cplx::complex<T>{2.02, 4.42},
          sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},
          sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},
          sycl::ext::cplx::complex<T>{inf_val<T>, inf_val<T>},
          sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},
          sycl::ext::cplx::complex<T>{inf_val<T>, inf_val<T>},
          sycl::ext::cplx::complex<T>{nan_val<T>, nan_val<T>},
      });
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input2[i] = sycl::ext::cplx::complex<T>{cplx_input1[i].real(),
                                                 cplx_input1[i].imag()};
  }

  sycl::marray<std::complex<T>, NumElements> std_out{};
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = detail::std_pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);

  sycl::free(d_cplx_out, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex pow cplx-deci overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // std::complex test cases
  const auto std_in1 =
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
  const auto std_in2 = GENERATE(init_deci(sycl::marray<T, NumElements>{
      1.0,
      4.42,
      -3,
      4.0,
      2.02,
      inf_val<T>,
      inf_val<T>,
      2.02,
      nan_val<T>,
      nan_val<T>,
      inf_val<T>,
      nan_val<T>,
      inf_val<T>,
      nan_val<T>,
  }));

  // sycl::complex test cases
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input1;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input1[i] =
        sycl::ext::cplx::complex<T>{std_in1[i].real(), std_in1[i].imag()};
  }
  sycl::marray<T, NumElements> cplx_input2;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input2[i] = std_in2[i];
  }

  sycl::marray<std::complex<T>, NumElements> std_out{};
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = detail::std_pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);

  sycl::free(d_cplx_out, Q);
}

TEMPLATE_TEST_CASE_SIG("Test marray complex pow deci-cplx overload", "[pow]",
                       ((typename T, std::size_t NumElements), T, NumElements),
                       (double, 14), (float, 14), (sycl::half, 14)) {
  sycl::queue Q;

  // std::complex test cases
  const auto std_in1 = GENERATE(init_deci(sycl::marray<T, NumElements>{
      1.0,
      4.42,
      -3,
      4.0,
      2.02,
      inf_val<T>,
      inf_val<T>,
      2.02,
      nan_val<T>,
      nan_val<T>,
      inf_val<T>,
      nan_val<T>,
      inf_val<T>,
      nan_val<T>,
  }));
  const auto std_in2 =
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
  sycl::marray<T, NumElements> cplx_input1;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input1[i] = std_in1[i];
  }
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input2;
  for (std::size_t i = 0; i < NumElements; ++i) {
    cplx_input2[i] =
        sycl::ext::cplx::complex<T>{std_in2[i].real(), std_in2[i].imag()};
  }

  sycl::marray<std::complex<T>, NumElements> std_out{};
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> h_cplx_out;
  auto d_cplx_out = sycl::malloc_device<
      sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

  // Get std::complex output
  for (std::size_t i = 0; i < NumElements; ++i)
    std_out[i] = detail::std_pow(std_in1[i], std_in2[i]);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       d_cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
     }).wait();
    Q.copy(d_cplx_out, &h_cplx_out, 1).wait();

    check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);
  }

  // Check cplx::complex output from host
  h_cplx_out = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  check_results(h_cplx_out, std_out, /*tol_multiplier*/ 3);

  sycl::free(d_cplx_out, Q);
}
