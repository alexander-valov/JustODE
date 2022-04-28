#pragma once

#include <type_traits>
#include <vector>
#include <stdexcept>
#include <cassert>

#include "TraitsHelper.hpp"

namespace JustODE {

namespace detail {

template<class MatrixType>
class LUPFactorization {

public:

    template<class InputType>
    LUPFactorization(const InputType& matrix)
        : LU_(type_traits::rows(matrix), type_traits::cols(matrix)),
          P_(type_traits::rows(matrix)),
          nb_rows_permutations_(0),
          is_initialized_(false)
    {
        Compute(matrix);
    }

    template<class InputType>
    LUPFactorization(InputType& matrix)
        : LU_(matrix),
          P_(type_traits::rows(matrix)),
          nb_rows_permutations_(0),
          is_initialized_(false)
    {
        Compute();
    }

    const MatrixType& LU() const {
        assert(is_initialized_ && "LUPFactorization is not initialized");
        return LU_;
    }

    const std::vector<int>& P() const {
        assert(is_initialized_ && "LUPFactorization is not initialized");
        return P_;
    }

    type_traits::MatrixElemType<MatrixType> Determinant() const {
        assert(is_initialized_ && "LUPFactorization is not initialized");

        int mult = (nb_rows_permutations_ % 2 == 0) ? 1 : -1;
        type_traits::MatrixElemType<MatrixType> det = LU_(0, 0);
        for (int i = 1; i < type_traits::rows(LU_); i++) {
            det *= LU_(i, i);
        }
        return mult * det;
    }

    template<class InputType>
    void Compute(const InputType& matrix) {
        LU_ = matrix;
        Compute();
    }

    void Compute() {
        static_assert(
            type_traits::has_mehod_parenthesis_v<MatrixType>,
            "The type `MatrixType` must provide parenthesis operator()(size_t, size_t) to read/write element's access."
        );

        int n = type_traits::rows(LU_);

        nb_rows_permutations_ = 0;

        // initialize row permutations matrix
        P_.resize(n);
        for (int i = 0; i < n; i++) {
            P_[i] = i;
        }

        // outer loop over diagonal pivots
        for (int i = 0; i < n - 1; i++) {
            // Find the largest pivot
            int maxPivot = i;
            for (int k = i + 1; k < n; k++) {
                if (std::abs(LU_(k, i)) > std::abs(LU_(i, i))) {
                    maxPivot = k;
                }
            }

            // Check for singularity
            if (LU_(maxPivot, i) == 0) {
                throw std::invalid_argument("Matrix is singular");
            }

            // Swap rows if needed
            if (maxPivot != i) {
                std::swap(P_[maxPivot], P_[i]);
                for (int k = i; k < n; k++) {
                    std::swap(LU_(i, k), LU_(maxPivot, k));
                }
                ++nb_rows_permutations_;
            }

            // Gaussian elimination
            for (int k = i + 1; k < n; k++) {
                LU_(k, i) /= LU_(i, i);
                for (int j = i + 1; j < n; j++) {
                    LU_(k, j) = LU_(k, j) - LU_(i, j) * LU_(k, i);
                }
            }
        }

        is_initialized_ = true;
    }

    template<class Container>
    Container Solve(const Container& rhs) const {
        int n = std::size(rhs);
        Container result = rhs;

        // forward substitution algorithm
        for (int i = 0; i < n; i++) {
            auto it_res = std::next(std::begin(result), i);
            auto it_rhs = std::next(std::begin(rhs), P_[i]);
            *it_res = *it_rhs;
            for (int k = 0; k < i; k++) {
                auto it_res_prev = std::next(std::begin(result), k);
                *it_res -= LU_(i, k) * (*it_res_prev);
            }
        }

        // back substitution algorithm
        for (int i = n - 1; i >= 0; i--) {
            auto it_res = std::next(std::begin(result), i);
            for (int k = i + 1; k < n; k++) {
                auto it_res_prev = std::next(std::begin(result), k);
                *it_res -= LU_(i, k) * (*it_res_prev);
            }
            *it_res /= LU_(i, i);
        }

        return result;
    }

protected:
    MatrixType LU_;
    std::vector<int> P_;
    int nb_rows_permutations_;
    bool is_initialized_;
};

}

}