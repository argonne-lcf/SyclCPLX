# SyclCPLX

Sycl complex library header-only. 


A sycl port of https://github.com/llvm/llvm-project/tree/main/libcxx/include/complex.
See `include/sycl_ext_complex.hpp` 

```c++
#include <sycl/sycl.hpp>
#include "sycl_ext_complex.hpp"
#include <complex>
#include <cassert>

bool almost_equal(std::complex<double> x, std::complex<double> y, int ulp) {
   return std::abs(x-y) <= std::numeric_limits<double>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<double>::min();
}

int main() {

  sycl::queue Q(sycl::gpu_selector{});

  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  std::complex<double> i00{0.2,0.5};
  std::complex<double> i01{0.2,0.3};
  std::complex<double> cpu_result = std::pow(i00,i01);
  auto* gpu_result = sycl::malloc_shared<sycl::ext::cplx::complex<double>>(1,Q);

  Q.single_task([=]() {
    //Using implicit cast from std::complex -> sycl::ext::cplx::complex
    gpu_result[0] = sycl::ext::cplx::pow<double>(i00, i01);
  }).wait();

  std::cout << "cpu_result " << cpu_result << std::endl;
  std::cout << "gpu_result" << gpu_result[0] << std::endl;
  //Using implicit cast from sycl::ext::cplx::complex ->  std::complex
  assert(almost_equal(cpu_result, gpu_result[0], 1));
  return 0;
}
```

## Tests

Testing is implemented with Catch2 and CMake. Catch2 is added as a git submodule inside the `vendor` directory.

Instructions to build and run tests for DPCPP are below:
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=$CXX_PATH -DCMAKE_CXX_FLAGS=-fsycl ..
make -j 8
ctest
```

## Conversions

The implicit conversion to and from `std::complex` works well for values, but
does not work for references and pointers (typically used for arrays in linear
algebra libraries), and it often does not work with template argument
deduction. It is therefore necessary to use `reinterpret_cast` in some cases.
See [hello_mkl.cpp](hello_mkl.cpp) for an example of using the oneMKL library
with `sycl::ext::cplx::complex`.

To simplify usage within an application, it is recommended to use the sycl type
everywhere when possible, even for host data. For cross-vendor applications, a
complex type alias can be defined in an application specific namespace. For
example, `myapp::complex<T>` can be a templated alias for
`sycl::ext::cplx::complex<T>` when running with a SYCL backend, and
`thrust::complex` for CUDA or HIP backends. For an example of this, see the
complex implementation in [gtensor](https://github.com/wdmapp/gtensor).
