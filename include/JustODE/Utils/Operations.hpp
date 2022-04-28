#pragma once

#include <cmath>
#include <numeric>
#include <algorithm>

#include "TraitsHelper.hpp"

namespace JustODE {

namespace detail {

    // Coefficient-wise 
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container AbsCwise(const Container& container) {
        Container result = container;
        for (auto& item : result) {
            item = std::abs(item);
        }
        return result;
    }

    /// Coefficient-wise maximum
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MaxCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            if (*first2 > *first1) {
                *first1 = *first2;
            }
        }
        return res;
    }

    /// Coefficient-wise summation
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container PlusCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 += *first2;
        }
        return res;
    }
    /// Coefficient-wise summation
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container PlusCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 += *first2;
        }
        return left;
    }

    /// Coefficient-wise subtraction
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MinusCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 -= *first2;
        }
        return res;
    }
    /// Coefficient-wise subtraction
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container MinusCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 -= *first2;
        }
        return left;
    }

    /// Coefficient-wise division
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container DivCwise(const Container& left, const Container& right) {
        Container res = left;
        auto first1 = std::begin(res);
        auto last1  = std::end(res);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 /= *first2;
        }
        return res;
    }
    /// Coefficient-wise division
    template<class Container, type_traits::IsRealContainer<Container> = true>
    Container DivCwise(Container&& left, const Container& right) {
        auto first1 = std::begin(left);
        auto last1  = std::end(left);
        auto first2 = std::begin(right);
        for (; first1 != last1; ++first1, (void)++first2) {
            *first1 /= *first2;
        }
        return left;
    }

    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const Container& container, const T& scalar) {
        Container res = container;
        for (auto& elem : res) { elem *= scalar; }
        return res;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const T& scalar, const Container& container) {
        return MultScalar(container, scalar);
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(Container&& container, const T& scalar) {
        for (auto& elem : container) { elem *= scalar; }
        return container;
    }
    /// Multiplies all coefficients by the given scalar
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container MultScalar(const T& scalar, Container&& container) {
        return MultScalar(container, scalar);
    }

    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const Container& container, const T& scalar) {
        Container res = container;
        for (auto& elem : res) { elem += scalar; }
        return res;
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const T& scalar, const Container& container) {
        return PlusScalar(container, scalar);
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(Container&& container, const T& scalar) {
        for (auto& elem : container) { elem += scalar; }
        return container;
    }
    /// Add the given scalar to all coefficients
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    Container PlusScalar(const T& scalar, Container&& container) {
        return PlusScalar(container, scalar);
    }

    /// Matrix-vector multiplication
    template<
        class MatrixType,
        class Container,
        type_traits::IsSupportsParenthesis<MatrixType> = true,
        type_traits::IsRealContainer<Container> = true
    >
    Container MVProd(const MatrixType& matrix, const Container& vec) {
        Container result = vec;
        for (auto& item : result) { item = 0; }
        for (int ic = 0; ic < type_traits::cols(matrix); ic++) {
            auto vec_it = std::next(std::begin(vec), ic);
            for (int ir = 0; ir < type_traits::rows(matrix); ir++) {
                auto res_it = std::next(std::begin(result), ir);
                *res_it += matrix(ir, ic) * (*vec_it);
            }
        }
        return result;
    }

    /// Computes Root Mean Square norm on the given container
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    T RMSNorm(const Container& container, T init) {
        return std::sqrt(
            std::transform_reduce(
                std::begin(container), std::end(container), std::begin(container), init
            ) / std::size(container)
        );
    }

    /// Squared 2-norm of the vector
    template<class Container, class T, type_traits::IsRealContainer<Container> = true>
    T SquaredNorm(const Container& container, T init) {
        return std::transform_reduce(
            std::begin(container), std::end(container), std::begin(container), init
        );
    }

}

}