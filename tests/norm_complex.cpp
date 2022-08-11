#include "test_helper.hpp"

template <typename T> struct test_norm {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re, init_im);
    sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};

    T std_out{};
    auto *cplx_out = sycl::malloc_shared<T>(1, Q);

    // Get std::complex output
    std_out = std::norm(std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::norm<T>(cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = sycl::ext::cplx::norm<T>(cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

template <typename T, std::size_t NumElements> struct test_norm_marray {
  bool operator()(sycl::queue &Q, test_marray<T, NumElements> init_re,
                  test_marray<T, NumElements> init_im) {
    bool pass = true;

    auto std_in = init_std_complex(init_re.get(), init_im.get());
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements> cplx_input =
        sycl::ext::cplx::make_complex_marray(init_re.get(), init_im.get());

    sycl::marray<T, NumElements> std_out{};
    auto *cplx_out = sycl::malloc_shared<sycl::marray<T, NumElements>>(1, Q);

    // Get std::complex output
    for (std::size_t i = 0; i < NumElements; ++i)
      std_out[i] = std::norm(std_in[i]);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       *cplx_out = sycl::ext::cplx::norm<T>(cplx_input);
     }).wait();

    pass &= check_results(*cplx_out, std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    *cplx_out = sycl::ext::cplx::norm<T>(cplx_input);

    pass &= check_results(*cplx_out, std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

// Difference between libstdc++ and libc++ when NaN's and Inf values are
// combined.
int main() {
  sycl::queue Q;

  bool test_passes = true;
  {
    test_passes &= test_valid_types<test_norm>(Q, 4.42, 2.02);

    test_passes &= test_valid_types<test_norm>(Q, INFINITY, 2.02);
    test_passes &= test_valid_types<test_norm>(Q, 4.42, INFINITY);
    test_passes &= test_valid_types<test_norm>(Q, INFINITY, INFINITY);

    test_passes &= test_valid_types<test_norm>(Q, NAN, 2.02);
    test_passes &= test_valid_types<test_norm>(Q, 4.42, NAN);
    test_passes &= test_valid_types<test_norm>(Q, NAN, NAN);
  }

  // marray tests
  {
    constexpr size_t m_size = 10;
    test_marray<double, m_size> re = {1.0,       4.42,      -3,   4.0,  2.02,
                                      INFINITYd, INFINITYd, 2.02, NANd, NANd};
    test_marray<double, m_size> im = {1.0,  2.02,      3.5,  -4.0, INFINITYd,
                                      4.42, INFINITYd, NANd, 4.42, NANd};
    test_passes &= test_valid_types<test_norm_marray, m_size>(Q, re, im);
  }

  if (!test_passes)
    std::cerr << "acos complex test fails\n";

  return !test_passes;
}
