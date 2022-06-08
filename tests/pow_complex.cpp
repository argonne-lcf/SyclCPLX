#include "test_helper.hpp"

template<typename T>
bool test_pow_cplx_cplx(sycl::queue &Q, T init_re, T init_im) {
  bool pass = true;

  auto std_in1 = init_std_complex(init_re, init_im);
  auto std_in2 = init_std_complex(init_re, init_im);
  sycl::ext::cplx::complex<T> cplx_input1{init_re, init_im};
  sycl::ext::cplx::complex<T> cplx_input2{init_re, init_im};

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_in1, std_in2);

  // Check cplx::complex output from device
  Q.single_task([=]() {
      cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);
    }).wait();

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow<T>(cplx_input1, cplx_input2);

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

  return pass;
}

template<typename T>
bool test_pow_cplx_deci(sycl::queue &Q, T init_re, T init_im) {
  bool pass = true;

  auto std_in = init_std_complex(init_re, init_im);
  auto std_deci_in = init_deci(init_re);
  sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};
  T deci_input = init_re;

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_in, std_deci_in);

  // Check cplx::complex output from device
  Q.single_task([=]() {
      cplx_out[0] = sycl::ext::cplx::pow(cplx_input, deci_input);
    }).wait();

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow(cplx_input, deci_input);

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

  return pass;
}

template<typename T>
bool test_pow_deci_cplx(sycl::queue &Q, T init_re, T init_im) {
  bool pass = true;

  auto std_in = init_std_complex(init_re, init_im);
  auto std_deci_in = init_deci(init_re);
  sycl::ext::cplx::complex<T> cplx_input{init_re, init_im};
  T deci_input = init_re;

  std::complex<T> std_out{};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  std_out = std::pow(std_deci_in, std_in);

  // Check cplx::complex output from device
  Q.single_task([=]() {
      cplx_out[0] = sycl::ext::cplx::pow(deci_input, cplx_input);
    }).wait();

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::pow(deci_input, cplx_input);

  pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

  return pass;
}

template <typename T> struct test_pow {
  bool operator()(sycl::queue &Q, T init_re, T init_im) {
    bool pass = true;
    pass &= test_pow_cplx_cplx(Q, init_re, init_im);
    pass &= test_pow_cplx_deci(Q, init_re, init_im);
    pass &= test_pow_deci_cplx(Q, init_re, init_im);
    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;
  test_passes &= test_valid_types<test_pow>(Q, 4.42,  2.02);
  
  test_passes &= test_valid_types<test_pow>(Q, INFINITY, 2.02);
  test_passes &= test_valid_types<test_pow>(Q, 4.42, INFINITY);
  test_passes &= test_valid_types<test_pow>(Q, INFINITY, INFINITY);

  test_passes &= test_valid_types<test_pow>(Q, NAN, 2.02);
  test_passes &= test_valid_types<test_pow>(Q, 4.42, NAN);
  test_passes &= test_valid_types<test_pow>(Q, NAN, NAN);

  test_passes &= test_valid_types<test_pow>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_pow>(Q, INFINITY, NAN);
  test_passes &= test_valid_types<test_pow>(Q, NAN, INFINITY);
  test_passes &= test_valid_types<test_pow>(Q, INFINITY, NAN);

  if (!test_passes)
    std::cerr << "pow complex test fails\n";
}
