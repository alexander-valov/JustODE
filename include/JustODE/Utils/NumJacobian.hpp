#pragma once

#include "Matrix.hpp"
#include "Operations.hpp"

#include <type_traits>

namespace JustODE {

namespace detail {


template<class Callable, class Container, class T>
Matrix<T> NumJacForward(Callable&& func, const Container& x, const T& h) {
    static_assert(
        std::is_invocable_r_v<Container, Callable&&, const Container&>,
        "Invalid signature or return type of the function for which the Jacobian calculation is required!"
    );

    std::size_t n = std::size(x);
    Matrix<T> jacobian(n, n);
    Container f = func(x);
    for (std::size_t ic = 0; ic < n; ic++) {
        Container x_new = x;
        auto arg_pos = std::next(std::begin(x_new), ic);
        *arg_pos += h;

        Container f_new = func(x_new);
        Container diff = MinusCwise(f_new, f);
        for (std::size_t ir = 0; ir < n; ir++) {
            auto diff_it = std::next(std::begin(diff), ir);
            jacobian(ir, ic) = *diff_it / h;
        }
    }
    return jacobian;
}

}

}