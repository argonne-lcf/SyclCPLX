#include "sycl_ext_complex.hpp"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#define __half_epsilon 9.765625e-04f
#define __half_min 6.103515625e-05f
using namespace std;
bool almost_equal(float x, float y, int ulp) { return std::abs(x - y) <= __half_epsilon * std::abs(x + y) * ulp || std::abs(x - y) < __half_min; }
bool almost_equal(complex<float> x, complex<float> y, int ulp) { return almost_equal(x.real(), y.real(), ulp) && almost_equal(x.imag(), y.imag(), ulp); }
complex<float> sycl_half_to_float(sycl::ext::cplx::complex<sycl::half> c) {
  auto c_sycl_float = static_cast<sycl::ext::cplx::complex<float>>(c);
  return static_cast<complex<float>>(c_sycl_float);
}
complex<float> trunc_float(complex<float> c) {
  auto c_sycl_half = static_cast<sycl::ext::cplx::complex<sycl::half>>(c);
  return sycl_half_to_float(c_sycl_half);
}
void test_asinh() {
  sycl::queue Q(sycl::gpu_selector{});
  const char *usr_precision = getenv("SYCL_TOL_ULP");
  const int precision = usr_precision ? atoi(usr_precision) : 1;
  complex<float> in0_host{0.42, 0.0};
  complex<float> in0_host_trunc = trunc_float(in0_host);
  sycl::ext::cplx::complex<sycl::half> in0_device{0.42, 0.0};
  complex<float> out1_host{};
  auto *out1_device = sycl::malloc_shared<sycl::ext::cplx::complex<sycl::half>>(1, Q);
  { out1_host = asinh(in0_host_trunc); }
  Q.single_task([=]() { out1_device[0] = sycl::ext::cplx::asinh<sycl::half>(in0_device); }).wait();
  {
    if (!almost_equal(trunc_float(out1_host), sycl_half_to_float(out1_device[0]), precision)) {
      std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10) << "Host: " << trunc_float(out1_host) << " GPU: " << sycl_half_to_float(out1_device[0]) << std::endl;
      std::exit(112);
    }
  }
}
int main() { test_asinh(); }
