#include "test_helper.hpp"

template <typename T> struct test_proj {
  bool test_cplx(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::proj(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::proj<T>(cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::proj<T>(cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool test_deci(sycl::queue &Q, T init_re) {
    bool pass = true;

    auto std_in = init_deci(init_re);

    std::complex<T> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

    // Get std::complex output
    std_out = std::proj(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::proj<T>(init_re);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::proj<T>(init_re);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    pass &= test_cplx(Q, init_re, init_im);
    pass &= test_deci(Q, init_re);

    return pass;
  }
};

template <typename T, std::size_t NumElements> struct test_proj_marray {
  bool test_cplx(sycl::queue &Q, test_marray<T, NumElements> init_re,
                 test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re.get(), init_im.get());
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =
        sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

    sycl::marray<std::complex<T>, NumElements> std_out{};
    auto *cplx_out = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Get std::complex output
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::proj(std_in[i]);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::proj<T>(cplx_input);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::proj<T>(cplx_input);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool test_deci(sycl::queue &Q, test_marray<T, NumElements> init_re) {
    bool pass = true;

    auto std_in = init_deci(init_re.get());
    sycl::marray<T, NumElements> cplx_input = init_re.get();

    sycl::marray<std::complex<T>, NumElements> std_out{};
    auto *cplx_out = sycl::malloc_shared<
        sycl::marray<sycl::ext::cplx::complex<T>, NumElements>>(1, Q);

    // Get std::complex output
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::proj(std_in[i]);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::proj(cplx_input);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::proj(cplx_input);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }

  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool pass = true;

    pass &= test_cplx(Q, init_re, init_im);
    pass &= test_deci(Q, init_re);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_proj>(Q, 4.42, 2.02);

  test_passes &= test_valid_types<test_proj>(Q, INFINITY, 2.02);
  test_passes &= test_valid_types<test_proj>(Q, 4.42, INFINITY);
  test_passes &= test_valid_types<test_proj>(Q, INFINITY, INFINITY);

  test_passes &= test_valid_types<test_proj>(Q, NAN, 2.02);
  test_passes &= test_valid_types<test_proj>(Q, 4.42, NAN);
  test_passes &= test_valid_types<test_proj>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_proj>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_proj>(Q, INFINITY, NAN);
  test_passes &= test_valid_types<test_proj>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_proj>(Q, INFINITY, NAN);

  // marray test
  constexpr size_t m_size = 4;
  test_marray<double, m_size> A = {1, 4.42, -3, 4};
  test_marray<double, m_size> B = {1, 2.02, 3.5, -4};
  test_passes &= test_valid_types<test_proj_marray, m_size>(Q, A, B);

  if (!test_passes)
    std::cerr << "proj complex test fails\n";

  return !test_passes;
}
