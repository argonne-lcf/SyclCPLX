#include "test_helper.hpp"

template <typename T>
struct test_stream_operator {
  bool operator() (sycl::queue &Q, T init_re, T init_im) {
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);
    cplx_out[0] = sycl::ext::cplx::complex<T>(init_re, init_im);

    Q.submit([&](sycl::handler &CGH) {
      sycl::stream Out(512, 20, CGH);
      CGH.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
        Out << cplx_out[idx] << sycl::endl;
      });
    });

    return true;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  test_passes &= test_valid_types<test_stream_operator>(Q, 0.42, 0);
  test_passes &= test_valid_types<test_stream_operator>(Q, INFINITY, INFINITY);
  test_passes &= test_valid_types<test_stream_operator>(Q, NAN, NAN);

  if (!test_passes)
    std::cerr << "Stream operator with complex test fails\n";

  return 0;
}
