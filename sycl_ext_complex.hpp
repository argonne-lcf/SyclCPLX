#include <sycl/sycl.hpp>
#include <complex>

namespace sycl
{
    namespace ext
    {

        template<typename _Tp>
        class complex
        {
        public:
        complex(const _Tp& __r = _Tp(), const _Tp& __i = _Tp())
        : __real_(__r), __imag_(__i) { }

        constexpr _Tp real() const { return __real_; }
        constexpr _Tp imag() const { return __imag_; }

        //template<typename _Up>
        //constexpr sycl::ext::complex<_Tp>& operator=(const sycl::ext::complex<_Up>&); 

        _Tp& operator+=(const _Tp& __z)
        {
            __real_ += __z.real();
            __imag_ += __z.imag();
            return *this;
        }
        
        private:
          _Tp __real_;
          _Tp __imag_;
        };

        template<class _Tp>
        sycl::ext::complex<_Tp>
        sinh(const sycl::ext::complex<_Tp>& __z)
        {
            const _Tp __x = __z.real();
            const _Tp __y = __z.imag();
            return sycl::ext::complex<_Tp>(sycl::sinh(__x) * sycl::cos(__y), sycl::cosh(__x) * sycl::sin(__y));
        }

        /* sin(z) = -i sinh(i z) */
        // https://github.com/ifduyue/musl/blob/master/src/complex/csin.c
        template<class _Tp>
        sycl::ext::complex<_Tp>
        sin(const sycl::ext::complex<_Tp>& __z)
        {
            const sycl::ext::complex<_Tp> __u = sycl::ext::sinh(sycl::ext::complex<_Tp>(- __z.imag(), __z.real()));
            return sycl::ext::complex<_Tp>(__u.imag(), -__u.real());
        }
    }
}
