#include "test_helper.hpp"

#define test_op(test_name, label, op)                                          \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    using T = TestType;                                                        \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T> input1 = GENERATE(                                                \
        cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},                      \
        cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},          \
        cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},                \
        cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},    \
        cmplx<T>{inf_val<T>, nan_val<T>});                                     \
                                                                               \
    cmplx<T> input2 = GENERATE(cmplx<T>{4.42, 2.02});                          \
                                                                               \
    auto std_in1 = init_std_complex(input1);                                   \
    auto std_in2 = init_std_complex(input2);                                   \
    sycl::ext::cplx::complex<T> cplx_input1{input1.re, input1.im};             \
    sycl::ext::cplx::complex<T> cplx_input2{input2.re, input2.im};             \
                                                                               \
    std::complex<T> std_out{};                                                 \
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);   \
                                                                               \
    /* Check complex-complex op */                                             \
    std_out = std_in1 op std_in2;                                              \
                                                                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() {                                                    \
         cplx_out[0] = cplx_input1 op cplx_input2;                             \
       }).wait();                                                              \
      check_results(cplx_out[0], std_out);                                     \
    }                                                                          \
                                                                               \
    cplx_out[0] = cplx_input1 op cplx_input2;                                  \
                                                                               \
    check_results(cplx_out[0], std_out);                                       \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_op("Test complex addition cplx-cplx overload", "[add]", +);
test_op("Test complex subtraction cplx-cplx overload", "[sub]", -);
test_op("Test complex multiplication cplx-cplx overload", "[mul]", *);
test_op("Test complex division cplx-cplx overload", "[div]", /);

#undef test_op

#define test_op(test_name, label, op)                                          \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    using T = TestType;                                                        \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T> input1 = GENERATE(                                                \
        cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},                      \
        cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},          \
        cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},                \
        cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},    \
        cmplx<T>{inf_val<T>, nan_val<T>});                                     \
                                                                               \
    T input2 = GENERATE(4.42, 2.02);                                           \
                                                                               \
    auto std_in = init_std_complex(input1);                                    \
    auto std_deci_in = init_deci(input2);                                      \
    sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};              \
    T deci_input = input2;                                                     \
                                                                               \
    std::complex<T> std_out{};                                                 \
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);   \
                                                                               \
    /* Check complex-decimal op */                                             \
    std_out = std_in op std_deci_in;                                           \
                                                                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { cplx_out[0] = cplx_input op deci_input; }).wait(); \
      check_results(cplx_out[0], std_out);                                     \
    }                                                                          \
                                                                               \
    cplx_out[0] = cplx_input op deci_input;                                    \
                                                                               \
    check_results(cplx_out[0], std_out);                                       \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_op("Test complex addition cplx-deci overload", "[add]", +);
test_op("Test complex subtraction cplx-deci overload", "[sub]", -);
test_op("Test complex multiplication cplx-deci overload", "[mul]", *);
test_op("Test complex division cplx-deci overload", "[div]", /);

#undef test_op

#define test_op(test_name, label, op)                                          \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    using T = TestType;                                                        \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T> input1 = GENERATE(                                                \
        cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},                      \
        cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},          \
        cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},                \
        cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},    \
        cmplx<T>{inf_val<T>, nan_val<T>});                                     \
                                                                               \
    T input2 = GENERATE(4.42, 2.02);                                           \
                                                                               \
    auto std_in = init_std_complex(input1);                                    \
    auto std_deci_in = init_deci(input2);                                      \
    sycl::ext::cplx::complex<T> cplx_input{input1.re, input1.im};              \
    T deci_input = input2;                                                     \
                                                                               \
    std::complex<T> std_out{};                                                 \
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);   \
                                                                               \
    /* Check complex-decimal op */                                             \
    std_out = std_deci_in op std_in;                                           \
                                                                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { cplx_out[0] = deci_input op cplx_input; }).wait(); \
                                                                               \
      check_results(cplx_out[0], std_out);                                     \
    }                                                                          \
                                                                               \
    cplx_out[0] = deci_input op cplx_input;                                    \
                                                                               \
    check_results(cplx_out[0], std_out);                                       \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_op("Test complex addition deci-cplx overload", "[add]", +);
test_op("Test complex subtraction deci-cplx overload", "[sub]", -);
test_op("Test complex multiplication deci-cplx overload", "[mul]", *);
test_op("Test complex division deci-cplx overload", "[div]", /);

#undef test_op

// OP assign tests are checked for the all possible type combinations as op
// assign supports different types being used.

#define test_op_assign(test_name, label, op_assign)                            \
  TEMPLATE_TEST_CASE(                                                          \
      test_name, label, (std::tuple<double, double>),                          \
      (std::tuple<double, float>), (std::tuple<double, sycl::half>),           \
      (std::tuple<float, double>), (std::tuple<float, float>),                 \
      (std::tuple<float, sycl::half>), (std::tuple<sycl::half, double>),       \
      (std::tuple<sycl::half, float>), (std::tuple<sycl::half, sycl::half>)) { \
    using T1 = typename std::tuple_element<0, TestType>::type;                 \
    using T2 = typename std::tuple_element<1, TestType>::type;                 \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T1> input1 = GENERATE(                                               \
        cmplx<T1>{4.42, 2.02}, cmplx<T1>{inf_val<T1>, 2.02},                   \
        cmplx<T1>{4.42, inf_val<T1>}, cmplx<T1>{inf_val<T1>, inf_val<T1>},     \
        cmplx<T1>{nan_val<T1>, 2.02}, cmplx<T1>{4.42, nan_val<T1>},            \
        cmplx<T1>{nan_val<T1>, nan_val<T1>},                                   \
        cmplx<T1>{nan_val<T1>, inf_val<T1>},                                   \
        cmplx<T1>{inf_val<T1>, nan_val<T1>});                                  \
                                                                               \
    cmplx<T2> input2 = GENERATE(                                               \
        cmplx<T2>{2.02, 4.42}, cmplx<T2>{inf_val<T2>, 2.02},                   \
        cmplx<T2>{4.42, inf_val<T2>}, cmplx<T2>{inf_val<T2>, inf_val<T2>},     \
        cmplx<T2>{nan_val<T2>, 2.02}, cmplx<T2>{4.42, nan_val<T2>},            \
        cmplx<T2>{nan_val<T2>, nan_val<T2>},                                   \
        cmplx<T2>{nan_val<T2>, inf_val<T2>},                                   \
        cmplx<T2>{inf_val<T2>, nan_val<T2>});                                  \
                                                                               \
    auto std_in = init_std_complex(input1);                                    \
    sycl::ext::cplx::complex<T1> cplx_input{input1.re, input1.im};             \
                                                                               \
    auto *cplx_inout =                                                         \
        sycl::malloc_shared<sycl::ext::cplx::complex<T2>>(1, Q);               \
                                                                               \
    auto std_inout = init_std_complex(input2);                                 \
    cplx_inout[0].real(input2.re);                                             \
    cplx_inout[0].imag(input2.im);                                             \
                                                                               \
    std_inout op_assign std_in;                                                \
                                                                               \
    if (is_type_supported<T1>(Q) && is_type_supported<T2>(Q)) {                \
      Q.single_task([=]() { cplx_inout[0] op_assign cplx_input; }).wait();     \
                                                                               \
      check_results(cplx_inout[0],                                             \
                    std::complex<T2>(std_inout.real(), std_inout.imag()));     \
    }                                                                          \
                                                                               \
    cplx_inout[0].real(input2.re);                                             \
    cplx_inout[0].imag(input2.im);                                             \
                                                                               \
    cplx_inout[0] op_assign cplx_input;                                        \
                                                                               \
    check_results(cplx_inout[0],                                               \
                  std::complex<T2>(std_inout.real(), std_inout.imag()));       \
                                                                               \
    sycl::free(cplx_inout, Q);                                                 \
  }

test_op_assign("Test complex assign addition cplx-cplx overload", "[add]", +=);
test_op_assign("Test complex assign subtraction cplx-cplx overload", "[sub]",
               -=);
test_op_assign("Test complex assign multiplication cplx-cplx overload", "[mul]",
               *=);
test_op_assign("Test complex assign division cplx-cplx overload", "[div]", /=);

#undef test_op_assign

#define test_op_assign(test_name, label, op_assign)                            \
  TEMPLATE_TEST_CASE(                                                          \
      test_name, label, (std::tuple<double, double>),                          \
      (std::tuple<double, float>), (std::tuple<double, sycl::half>),           \
      (std::tuple<float, double>), (std::tuple<float, float>),                 \
      (std::tuple<float, sycl::half>), (std::tuple<sycl::half, double>),       \
      (std::tuple<sycl::half, float>), (std::tuple<sycl::half, sycl::half>)) { \
    using T1 = typename std::tuple_element<0, TestType>::type;                 \
    using T2 = typename std::tuple_element<1, TestType>::type;                 \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    T1 input1 = GENERATE(4.42, 2.02, inf_val<T2>, nan_val<T2>);                \
                                                                               \
    cmplx<T2> input2 = GENERATE(                                               \
        cmplx<T2>{4.42, 2.02}, cmplx<T2>{inf_val<T2>, 2.02},                   \
        cmplx<T2>{4.42, inf_val<T2>}, cmplx<T2>{inf_val<T2>, inf_val<T2>},     \
        cmplx<T2>{nan_val<T2>, 2.02}, cmplx<T2>{4.42, nan_val<T2>},            \
        cmplx<T2>{nan_val<T2>, nan_val<T2>},                                   \
        cmplx<T2>{nan_val<T2>, inf_val<T2>},                                   \
        cmplx<T2>{inf_val<T2>, nan_val<T2>});                                  \
                                                                               \
    auto std_deci_in = init_deci(input1);                                      \
    T1 deci_input = input1;                                                    \
                                                                               \
    auto *cplx_inout =                                                         \
        sycl::malloc_shared<sycl::ext::cplx::complex<T2>>(1, Q);               \
                                                                               \
    auto std_inout = init_std_complex(input2);                                 \
    cplx_inout[0].real(input2.re);                                             \
    cplx_inout[0].imag(input2.im);                                             \
                                                                               \
    std_inout op_assign std_deci_in;                                           \
                                                                               \
    if (is_type_supported<T1>(Q) && is_type_supported<T2>(Q)) {                \
      Q.single_task([=]() { cplx_inout[0] op_assign deci_input; }).wait();     \
                                                                               \
      check_results(cplx_inout[0],                                             \
                    std::complex<T2>(std_inout.real(), std_inout.imag()));     \
    }                                                                          \
                                                                               \
    cplx_inout[0].real(input2.re);                                             \
    cplx_inout[0].imag(input2.im);                                             \
                                                                               \
    cplx_inout[0] op_assign deci_input;                                        \
                                                                               \
    check_results(cplx_inout[0],                                               \
                  std::complex<T2>(std_inout.real(), std_inout.imag()));       \
                                                                               \
    sycl::free(cplx_inout, Q);                                                 \
  }

test_op_assign("Test complex assign addition cplx-deci overload", "[add]", +=);
test_op_assign("Test complex assign subtraction cplx-deci overload", "[sub]",
               -=);
test_op_assign("Test complex assign multiplication cplx-deci overload", "[mul]",
               *=);
test_op_assign("Test complex assign division cplx-deci overload", "[div]", /=);

#undef test_op_assign

#define test_unary_op(test_name, label, op)                                    \
  TEMPLATE_TEST_CASE(test_name, label, double, float, sycl::half) {            \
    using T = TestType;                                                        \
    using std::make_tuple;                                                     \
                                                                               \
    sycl::queue Q;                                                             \
                                                                               \
    cmplx<T> input = GENERATE(                                                 \
        cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},                      \
        cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},          \
        cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},                \
        cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},    \
        cmplx<T>{inf_val<T>, nan_val<T>});                                     \
                                                                               \
    auto std_in = init_std_complex(input);                                     \
    sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};                \
                                                                               \
    std::complex<T> std_out{};                                                 \
    auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);   \
                                                                               \
    /* Check complex-decimal op */                                             \
    std_out = op std_in;                                                       \
                                                                               \
    if (is_type_supported<T>(Q)) {                                             \
      Q.single_task([=]() { cplx_out[0] = op cplx_input; }).wait();            \
                                                                               \
      check_results(cplx_out[0], std_out);                                     \
    }                                                                          \
                                                                               \
    cplx_out[0] = op cplx_input;                                               \
                                                                               \
    check_results(cplx_out[0], std_out);                                       \
                                                                               \
    sycl::free(cplx_out, Q);                                                   \
  }

test_unary_op("Test complex addition unary operator", "[add]", +);
test_unary_op("Test complex subtraction unary operator", "[sub]", -);

#undef test_unary_op
