#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <sstream>

#include <iostream>

namespace JustODE {

namespace detail {

template<class T>
class Matrix {

public:
    Matrix(std::size_t nrows, std::size_t ncols)
      : n_rows_(nrows), n_cols_(ncols), data_(nrows * ncols) {}

    Matrix(std::size_t nrows, std::size_t ncols, const T* data_col_major)
      : n_rows_(nrows), n_cols_(ncols), data_(data_col_major, data_col_major + nrows * ncols) {}

    Matrix(std::size_t nrows, std::size_t ncols, const std::vector<T>& data_col_major)
      : n_rows_(nrows), n_cols_(ncols), data_(data_col_major) {}

    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        std::vector<std::size_t> col_sizes;
        for (const auto& row : list) {
            col_sizes.push_back(row.size());
        }
        bool is_cols_equals_and_positive = std::all_of(
            col_sizes.begin(), col_sizes.end(), [&](std::size_t i){ return (i == col_sizes[0]) && (i > 0); }
        );
        assert(is_cols_equals_and_positive && "Incorrect matrix format (initializer list)");

        n_rows_ = list.size();
        if (list.size() > 0) {
            n_cols_ = col_sizes.front();
            data_.resize(n_rows_ * n_cols_);
            
            auto it_rows = list.begin();
            for (std::size_t ir = 0; ir < n_rows_; ++ir, ++it_rows) {
                auto it_cols = it_rows->begin();
                for (std::size_t ic = 0; ic < n_cols_; ++ic, ++it_cols) {
                    operator()(ir, ic) = *it_cols;
                }
            }
        } else {
            n_cols_ = 0;
        }
    }

    std::size_t rows() const { return n_rows_; }
    std::size_t cols() const { return n_cols_; }
    const std::vector<T>& data() const { return data_; }

    const T& operator()(std::size_t row, std::size_t col) const {
        return data_[row + n_rows_ * col];
    }
    T& operator()(std::size_t row, std::size_t col) {
        return data_[row + n_rows_ * col];
    }

    const T& at(std::size_t row, std::size_t col) const {
        return data_.at(row + n_rows_ * col);
    }
    T& at(std::size_t row, std::size_t col) {
        return data_.at(row + n_rows_ * col);
    }

    template<class Scalar>
    struct comma_helper {
        comma_helper(Matrix<T>& m, const Scalar& value)
          : mat(m), row_(0), col_(1)
        {
            assert(mat.rows() > 0 && mat.cols() > 0
              && "Cannot initialize a 0x0 matrix (operator<<)");
            mat(0, 0) = value;
        }
        ~comma_helper() {
            assert((row_ + 1 == mat.rows() && col_ == mat.cols())
              && "Too few coefficients passed to comma initialization (operator<<)");
        }

        comma_helper& operator,(const Scalar& value) {
            if (col_ == mat.cols()) {
                row_ += 1;
                col_ = 0;
                assert(row_ < mat.rows() 
                  && "Too many rows passed to comma initialization (operator<<)");
            }
            mat(row_, col_) = value;
            col_ += 1;
            return *this;
        }

        Matrix<T>& mat;
        std::size_t row_;
        std::size_t col_;
    };

    comma_helper<T> operator<<(const T& value) {
        return comma_helper<T>(*this, value);
    }

    Matrix<T> transpose() const {
        Matrix<T> result(n_cols_, n_rows_);
        for (std::size_t ir = 0; ir < n_rows_; ir++) {
            for (std::size_t ic = 0; ic < n_cols_; ic++) {
                result(ic, ir) = this->operator()(ir, ic);
            }
        }
        return result;
    }

    bool operator==(const Matrix<T>& rhs) {
        return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && data_ == rhs.data_;
    }
    bool operator!=(const Matrix<T>& rhs) {
        return !(*this == rhs);
    }

    Matrix<T>& operator+=(const Matrix<T>& rhs) {
        assert((n_rows_ == rhs.n_rows_ || n_cols_ == rhs.n_cols_)
          && "The shape of the matrices does not match (operator+=)");
        std::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(), std::plus<>());
        return *this;
    }
    Matrix<T>& operator+=(Matrix<T>&& rhs) {
        assert((n_rows_ == rhs.n_rows_ || n_cols_ == rhs.n_cols_)
          && "The shape of the matrices does not match (operator+=)");
        std::transform(data_.begin(), data_.end(), std::make_move_iterator(rhs.data_.begin()), data_.begin(), std::plus<>());
        return *this;
    }

    Matrix<T>& operator-=(const Matrix<T>& rhs) {
        assert((n_rows_ == rhs.n_rows_ || n_cols_ == rhs.n_cols_)
          && "The shape of the matrices does not match (operator-=)");
        std::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(), std::minus<>());
        return *this;
    }
    Matrix<T>& operator-=(Matrix<T>&& rhs) {
        assert((n_rows_ == rhs.n_rows_ || n_cols_ == rhs.n_cols_)
          && "The shape of the matrices does not match (operator-=)");
        std::transform(data_.begin(), data_.end(), std::make_move_iterator(rhs.data_.begin()), data_.begin(), std::minus<>());
        return *this;
    }

    Matrix<T>& operator*=(const T& scalar) {
        std::transform(
            data_.begin(), data_.end(), data_.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, scalar)
        );
        return *this;
    }

protected:
    std::size_t n_rows_;
    std::size_t n_cols_;
    std::vector<T> data_;
};

template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    std::size_t max_width = 0;
    for (std::size_t ic = 0; ic < mat.cols(); ic++) {
        for (std::size_t ir = 0; ir < mat.rows(); ir++) {
            std::stringstream ss;
            ss << mat(ir, ic);
            max_width = std::max<std::size_t>(max_width, ss.str().size());
        }
    }

    for (std::size_t ir = 0; ir < mat.rows(); ir++) {
        for (std::size_t ic = 0; ic < mat.cols(); ic++) {
            if (ic > 0) { os << "  "; }
            os.width(max_width);
            os.fill(' ');
            os << mat(ir, ic);
        }
        os << "\n";
    }
    return os;
}

// --------------------------------------------------------------------
// Matrix addition operators
// --------------------------------------------------------------------
template<class T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> result(lhs);
    return (result += rhs);
}
template<class T>
Matrix<T> operator+(Matrix<T>&& lhs, const Matrix<T>& rhs) { return (lhs += rhs); }
template<class T>
Matrix<T> operator+(const Matrix<T>& lhs, Matrix<T>&& rhs) { return (rhs += lhs); }
template<class T>
Matrix<T> operator+(Matrix<T>&& lhs, Matrix<T>&& rhs) { return (rhs += lhs); }

// --------------------------------------------------------------------
// Matrix subtraction operators
// --------------------------------------------------------------------
template<class T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> result(lhs);
    return (result -= rhs);
}
template<class T>
Matrix<T> operator-(Matrix<T>&& lhs, const Matrix<T>& rhs) { return (lhs -= rhs); }
template<class T>
Matrix<T> operator-(const Matrix<T>& lhs, Matrix<T>&& rhs) { return (rhs -= lhs); }
template<class T>
Matrix<T> operator-(Matrix<T>&& lhs, Matrix<T>&& rhs) { return (rhs -= lhs); }

// --------------------------------------------------------------------
// Matrix multiplicaton by scalar
// --------------------------------------------------------------------
template<class T>
Matrix<T> operator*(const Matrix<T>& mat, const T& scalar) {
    Matrix<T> result(mat);
    return (result *= scalar);
}
template<class T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& mat) { return mat * scalar; }
template<class T>
Matrix<T> operator*(Matrix<T>&& mat, const T& scalar) { return (mat *= scalar); }
template<class T>
Matrix<T> operator*(const T& scalar, Matrix<T>&& mat) { return (mat *= scalar); }

// --------------------------------------------------------------------
// Matrix multiplicaton
// --------------------------------------------------------------------
template<class T>
Matrix<T> mat_mul_rect(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> result(lhs.rows(), rhs.cols());
    for (std::size_t ir = 0; ir < lhs.rows(); ir++) {
        // Save ir-th row of the lhs matrix
        std::vector<T> row;
        row.reserve(lhs.cols());
        for (std::size_t ic = 0; ic < lhs.cols(); ic++) {
            row.emplace_back(lhs(ir, ic));
        }
        // Calculate the ir-th row of the multiplication
        for (std::size_t ic = 0; ic < rhs.cols(); ic++) {
            result(ir, ic) = std::transform_reduce(row.begin(), row.end(), std::next(rhs.data().begin(), ic * rhs.rows()), T(0));
        }
    }
    return result;
}
template<class T>
Matrix<T> mat_mul_square_rl(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    for (std::size_t ir = 0; ir < lhs.rows(); ir++) {
        // Save ir-th row of the lhs matrix
        std::vector<T> row;
        row.reserve(lhs.cols());
        for (std::size_t ic = 0; ic < lhs.cols(); ic++) {
            row.emplace_back(lhs(ir, ic));
        }
        // Calculate the ir-th row of the multiplication
        for (std::size_t ic = 0; ic < rhs.cols(); ic++) {
            lhs(ir, ic) = std::transform_reduce(row.begin(), row.end(), std::next(rhs.data().begin(), ic * rhs.rows()), T(0));
        }
    }
    return std::move(lhs);
}
template<class T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    assert((lhs.cols() == rhs.rows())
      && "The matrix dimensions are inconsistent (operator*)");
    return mat_mul_rect(lhs, rhs);
}
template<class T>
Matrix<T> operator*(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    assert((lhs.cols() == rhs.rows())
      && "The matrix dimensions are inconsistent (operator*)");
    if (lhs.rows() == rhs.cols()) {
        return mat_mul_square_rl(std::forward<Matrix<T>>(lhs), rhs);
    } else {
        return mat_mul_rect(lhs, rhs);
    }
}
template<class T>
Matrix<T> operator*(const Matrix<T>& lhs, Matrix<T>&& rhs) {
    assert((lhs.cols() == rhs.rows())
      && "The matrix dimensions are inconsistent (operator*)");
    return mat_mul_rect(lhs, rhs);
}
template<class T>
Matrix<T> operator*(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    assert((lhs.cols() == rhs.rows())
      && "The matrix dimensions are inconsistent (operator*)");
    if (lhs.rows() == rhs.cols()) {
        return mat_mul_square_rl(std::forward<Matrix<T>>(lhs), rhs);
    } else {
        return mat_mul_rect(lhs, rhs);
    }
}

}

}