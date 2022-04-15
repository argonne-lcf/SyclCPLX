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
bool almost_equal(complex<double> x, complex<double> y, int ulp) {
   return std::abs(x-y) <= std::numeric_limits<double>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<double>::min();
}
void test_tanh(){
   sycl::queue Q(sycl::gpu_selector{});
   const char* usr_precision = getenv("SYCL_TOL_ULP");
   const int precision = usr_precision ? atoi(usr_precision) : 4;
   complex<double> in0_host { 0.42, 0.0 };
   complex<double> out1_host {};
   auto* out1_device = sycl::malloc_shared<sycl::ext::cplx::complex<double>>(1,Q);;
   {
      out1_host = tanh(in0_host);
   }
  Q.single_task([=]() {
   out1_device[0] = sycl::ext::cplx::tanh<double>(in0_host);
   }).wait();
   {
      if ( !almost_equal(out1_host,out1_device[0], precision) ) {
          std::cerr << std::setprecision (std::numeric_limits<double>::max_digits10 )
                    << "Host: " << out1_host << " GPU: " << out1_device[0] << std::endl;
          std::exit(112);
      }
   }
}
int main()
{
    test_tanh();
}
