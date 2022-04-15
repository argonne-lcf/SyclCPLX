#include <complex>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "sycl_ext_complex.hpp"
using namespace std;
bool almost_equal(complex<float> x, complex<float> y, int ulp) {
   return std::abs(x-y) <= std::numeric_limits<float>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<float>::min();
}
void test_pow(){
   sycl::queue Q(sycl::gpu_selector{});
   const char* usr_precision = getenv("SYCL_TOL_ULP");
   const int precision = usr_precision ? atoi(usr_precision) : 4;
   complex<float> in0_host { 0.42, 0.0 };
   sycl::ext::cplx::complex<float> in0_device { 0.42, 0.0 };
   complex<float> in1_host { 0.42, 0.0 };
   sycl::ext::cplx::complex<float> in1_device { 0.42, 0.0 };
   complex<float> out2_host {};
   auto* out2_device = sycl::malloc_shared<sycl::ext::cplx::complex<float>>(1,Q);;
   {
      out2_host = pow(in0_host, in1_host);
   }
  Q.single_task([=]() {
   out2_device[0] = sycl::ext::cplx::pow(in0_device, in1_device);
   }).wait();
   std::complex<float> out2_device_std = { out2_device[0].real(), out2_device[0].imag() };
   {
      if ( !almost_equal(out2_host,out2_device_std, precision) ) {
          std::cerr << std::setprecision (std::numeric_limits<float>::max_digits10 )
                    << "Host: " << out2_host << " GPU: " << out2_device_std << std::endl;
          std::exit(112);
      }
   }
}
int main()
{
    test_pow();
}
