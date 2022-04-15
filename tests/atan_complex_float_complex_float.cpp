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
void test_atan(){
   sycl::queue Q(sycl::gpu_selector{});
   const char* usr_precision = getenv("SYCL_TOL_ULP");
   const int precision = usr_precision ? atoi(usr_precision) : 4;
   complex<float> in0_host { 0.42, 0.0 };
   auto* out1_device = sycl::malloc_shared<sycl::ext::cplx::complex<float>>(1,Q);;
  Q.single_task([=]() {
   out1_device[0] = sycl::ext::cplx::atan<float>(in0_host);
   }).wait();
   {
     if ( !almost_equal(tan(out1_device[0]), in0_host, 2*precision) ) {
          std::cerr << std::setprecision (std::numeric_limits<float>::max_digits10 )
                    << "Expected:" << in0_host << " Got: " << tan(out1_device[0]) << std::endl;
          std::exit(112);
     }
   }
}
int main()
{
    test_atan();
}
