#include "test_helper.hpp"

template <typename T> struct test_sycl_stream_operator {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);
    cplx_out[0] = sycl::ext::cplx::complex<T>(init_re, init_im);

    Q.submit([&](sycl::handler &CGH) {
      sycl::stream Out(512, 20, CGH);
      CGH.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
        Out << cplx_out[idx] << sycl::endl;
      });
    });
    Q.wait();

    sycl::free(cplx_out, Q);

    return true;
  }
};

// Host only tests for std::basic_ostream and std::basic_istream
template <typename T> struct test_ostream_operator {
  bool operator()(T init_re, T init_im) {
    sycl::ext::cplx::complex<T> c(init_re, init_im);

    std::ostringstream os;
    os << c;

    std::ostringstream ref_oss;
    ref_oss << std::complex<T>(init_re, init_im);

    if (ref_oss.str() == os.str())
      return true;
    return false;
  }
};

template <typename T> struct test_istream_operator {
  bool operator()(T init_re, T init_im) {
    sycl::ext::cplx::complex<T> c(init_re, init_im);

    std::ostringstream ref_oss;
    ref_oss << "(" << init_re << "," << init_im << ")";

    std::istringstream iss(ref_oss.str());

    iss >> c;

    return check_results(c, std::complex<T>(init_re, init_im),
                         /*is_device*/ false);
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  test_passes &= test_valid_types<test_sycl_stream_operator>(Q, 0.42, -1.1);
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, INFINITY, INFINITY);
  test_passes &= test_valid_types<test_sycl_stream_operator>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_ostream_operator>(0.42, -1.1);
  test_passes &= test_valid_types<test_ostream_operator>(INFINITY, INFINITY);
  test_passes &= test_valid_types<test_ostream_operator>(NAN, NAN);

  test_passes &= test_valid_types<test_istream_operator>(0.42, -1.1);
  test_passes &= test_valid_types<test_istream_operator>(INFINITY, INFINITY);
  test_passes &= test_valid_types<test_istream_operator>(NAN, NAN);

  if (!test_passes)
    std::cerr << "Stream operator with complex test fails\n";

  return 0;
}
