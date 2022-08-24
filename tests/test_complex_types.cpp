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

TEMPLATE_TEST_CASE("Test complex abs types", "[abs]", double, float,
                   sycl::half) {
  static_assert(std::is_same_v<TestType, decltype(abs(complex<TestType>()))>);
}

TEMPLATE_TEST_CASE("Test complex polar types", "[abs]", double, float,
                   sycl::half) {
  static_assert(std::is_same_v<complex<TestType>, decltype(polar(TestType()))>);
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(polar(TestType(), TestType()))>);
}

TEMPLATE_TEST_CASE("Test complex pow types", "[abs]", double, float,
                   sycl::half) {
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(pow(complex<TestType>(), TestType()))>);
  static_assert(
      std::is_same_v<complex<TestType>,
                     decltype(pow(complex<TestType>(), complex<TestType>()))>);
  static_assert(std::is_same_v<complex<TestType>,
                               decltype(pow(TestType(), complex<TestType>()))>);
}
