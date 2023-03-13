#include "test_helper.hpp"

using namespace sycl::ext::cplx;

// Test checks user interface return types
// Compile time only tests, will fail during compilation due to static asserts

// Define math operations tests
#define TEST_MATH_OP_TYPE(test_name, label, op)                                \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
                                                                               \
    static_assert(std::is_same_v<complex<TestType>,                            \
                                 decltype(std::declval<complex<TestType>>()    \
                                              op std::declval<TestType>())>);  \
                                                                               \
    static_assert(                                                             \
        std::is_same_v<complex<TestType>,                                      \
                       decltype(std::declval<TestType>()                       \
                                    op std::declval<complex<TestType>>())>);   \
  }

TEST_MATH_OP_TYPE("Test complex addition types", "[add]", +)
TEST_MATH_OP_TYPE("Test complex subtraction types", "[sub]", -)
TEST_MATH_OP_TYPE("Test complex multiplication types", "[mul]", *)
TEST_MATH_OP_TYPE("Test complex division types", "[div]", /)
#undef TEST_MATH_FUNC_TYPE

// Define math function tests
#define TEST_MATH_FUNC_TYPE(test_name, label, func)                            \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    static_assert(std::is_same_v<complex<TestType>,                            \
                                 decltype(func(complex<TestType>()))>);        \
  }

TEST_MATH_FUNC_TYPE("Test complex acos types", "[acos]", acos)
TEST_MATH_FUNC_TYPE("Test complex asin types", "[asin]", asin)
TEST_MATH_FUNC_TYPE("Test complex atan types", "[atan]", atan)
TEST_MATH_FUNC_TYPE("Test complex acosh types", "[acosh]", acosh)
TEST_MATH_FUNC_TYPE("Test complex asinh types", "[asinh]", asinh)
TEST_MATH_FUNC_TYPE("Test complex atanh types", "[atanh]", atanh)
TEST_MATH_FUNC_TYPE("Test complex conj types", "[conj]", conj)
TEST_MATH_FUNC_TYPE("Test complex cos types", "[cos]", cos)
TEST_MATH_FUNC_TYPE("Test complex cosh types", "[cosh]", cosh)
TEST_MATH_FUNC_TYPE("Test complex exp types", "[exp]", exp)
TEST_MATH_FUNC_TYPE("Test complex log types", "[log]", log)
TEST_MATH_FUNC_TYPE("Test complex log10 types", "[log10]", log10)
TEST_MATH_FUNC_TYPE("Test complex proj types", "[proj]", proj)
TEST_MATH_FUNC_TYPE("Test complex sin types", "[sin]", sin)
TEST_MATH_FUNC_TYPE("Test complex sinh types", "[sinh]", sinh)
TEST_MATH_FUNC_TYPE("Test complex sqrt types", "[sqrt]", sqrt)
TEST_MATH_FUNC_TYPE("Test complex tan types", "[tan]", tan)
TEST_MATH_FUNC_TYPE("Test complex tanh types", "[tanh]", tanh)
#undef TEST_MATH_FUNC_TYPE

TEMPLATE_TEST_CASE("Test complex polar types", "[polar]", double, float,
                   sycl::half) {
  static_assert(std::is_same_v<complex<TestType>, decltype(polar(TestType()))>);
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(polar(TestType(), TestType()))>);
}

TEMPLATE_TEST_CASE("Test complex pow types", "[pow]", double, float,
                   sycl::half) {
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(pow(complex<TestType>(), TestType()))>);
  static_assert(
      std::is_same_v<complex<TestType>,
                     decltype(pow(complex<TestType>(), complex<TestType>()))>);
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(pow(TestType(), complex<TestType>()))>);
}

// Test F(Complex<T>) -> T
#define TEST_FUNC_TYPE(test_name, label, func)                                 \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    static_assert(                                                             \
        std::is_same_v<TestType, decltype(func(complex<TestType>()))>);        \
  }

TEST_FUNC_TYPE("Test complex abs deci - cplx types", "[abs]", abs)
TEST_FUNC_TYPE("Test complex real deci - cplx types", "[real]", real)
TEST_FUNC_TYPE("Test complex imag deci - cplx  types", "[imag]", imag)
#undef TEST_FUNC_TYPE

// Test F(T) -> T
#define TEST_FUNC_TYPE(test_name, label, func)                                 \
  TEMPLATE_TEST_CASE(test_name, label, (std::pair<double, bool>),              \
                     (std::pair<double, char>), (std::pair<double, int>),      \
                     (std::pair<sycl::half, sycl::half>),                      \
                     (std::pair<double, double>), (std::pair<float, float>)) { \
    using T = typename TestType::first_type;                                   \
    using X = typename TestType::second_type;                                  \
                                                                               \
    static_assert(std::is_same_v<T, decltype(func(X()))>);                     \
  }

TEST_FUNC_TYPE("Test complex arg deci - deci types", "[arg]", arg)
TEST_FUNC_TYPE("Test complex norm deci - deci types", "[norm]", norm)
TEST_FUNC_TYPE("Test complex real deci - deci types", "[real]", real)
TEST_FUNC_TYPE("Test complex imag deci - deci types", "[imag]", imag)
#undef TEST_FUNC_TYPE

// Test F(T) -> complex<T>
#define TEST_FUNC_TYPE(test_name, label, func)                                 \
  TEMPLATE_TEST_CASE(test_name, label, (std::pair<double, bool>),              \
                     (std::pair<double, char>), (std::pair<double, int>),      \
                     (std::pair<sycl::half, sycl::half>),                      \
                     (std::pair<double, double>), (std::pair<float, float>)) { \
    using T = typename TestType::first_type;                                   \
    using X = typename TestType::second_type;                                  \
                                                                               \
    static_assert(std::is_same_v<complex<T>, decltype(func(X()))>);            \
  }

TEST_FUNC_TYPE("Test complex conj cplx - deci types", "[conj]", conj)
TEST_FUNC_TYPE("Test complex proj cplx - deci types", "[proj]", proj)
#undef TEST_FUNC_TYPE

TEST_CASE("Test std::complex to sycl::complex conversion", "[conversion]") {
  auto complex_f = sycl::ext::cplx::complex<float>{42.f, 42.f};
  auto complex_d = sycl::ext::cplx::complex<double>{42.f, 42.f};
  auto complex_h = sycl::ext::cplx::complex<sycl::half>{42.f, 42.f};

  {
    auto f = static_cast<std::complex<float>>(complex_f);
    auto d = static_cast<std::complex<double>>(complex_f);
    auto h = static_cast<std::complex<sycl::half>>(complex_f);
  }
  {
    auto f = static_cast<std::complex<float>>(complex_d);
    auto d = static_cast<std::complex<double>>(complex_d);
    auto h = static_cast<std::complex<sycl::half>>(complex_d);
  }
  {
    auto f = static_cast<std::complex<float>>(complex_h);
    auto d = static_cast<std::complex<double>>(complex_h);
    auto h = static_cast<std::complex<sycl::half>>(complex_h);
  }
}

TEST_CASE("Test sycl::complex constructor from std::complex", "[constructor]") {
  auto complex_f = std::complex<float>{42.f, 42.f};
  auto complex_d = std::complex<double>{42.f, 42.f};
  auto complex_h = std::complex<sycl::half>{42.f, 42.f};

  {
    sycl::ext::cplx::complex<float> f{complex_f};
    sycl::ext::cplx::complex<double> d{complex_f};
    sycl::ext::cplx::complex<sycl::half> h{complex_f};
  }
  {
    sycl::ext::cplx::complex<float> f{complex_d};
    sycl::ext::cplx::complex<double> d{complex_d};
    sycl::ext::cplx::complex<sycl::half> h{complex_d};
  }
  {
    sycl::ext::cplx::complex<float> f{complex_h};
    sycl::ext::cplx::complex<double> d{complex_h};
    sycl::ext::cplx::complex<sycl::half> h{complex_h};
  }
}
