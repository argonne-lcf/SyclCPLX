#include "test_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

TEMPLATE_TEST_CASE("Test complex sycl stream", "[sycl::stream]", double, float,
                   sycl::half) {
  using T = TestType;

  sycl::queue Q;

  cmplx<T> input = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);
  cplx_out[0] = sycl::ext::cplx::complex<T>(input.re, input.im);

  Q.submit([&](sycl::handler &CGH) {
    sycl::stream Out(512, 20, CGH);
    CGH.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
      Out << cplx_out[idx] << sycl::endl;
    });
  }).wait();

  sycl::free(cplx_out, Q);
}

// Host only tests for std::basic_ostream and std::basic_istream
TEMPLATE_TEST_CASE("Test complex std ostream", "[ostream]", double, float,
                   sycl::half) {
  using T = TestType;

  sycl::queue Q;

  cmplx<T> input = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  sycl::ext::cplx::complex<T> c(input.re, input.im);

  std::ostringstream os;
  os << c;

  std::ostringstream ref_oss;
  ref_oss << std::complex<T>(input.re, input.im);

  CHECK(ref_oss.str() == os.str());
}

TEMPLATE_TEST_CASE("Test complex std istream", "[istream]", double, float,
                   sycl::half) {
  using T = TestType;

  sycl::queue Q;

  cmplx<T> input = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  sycl::ext::cplx::complex<T> c(input.re, input.im);

  std::ostringstream ref_oss;
  ref_oss << "(" << input.re << "," << input.im << ")";

  std::istringstream iss(ref_oss.str());

  iss >> c;

  check_results(c, std::complex<T>(input.re, input.im));
}
