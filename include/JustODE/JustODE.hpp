/*
 * JustODE https://github.com/alexander-valov/JustODE
 *
 * MIT License
 *
 * Copyright (c) 2022 Alexander Valov <https://github.com/alexander-valov>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "EmbeddedSolvers.hpp"

namespace JustODE {

/********************************************************************
 * @brief List of available methods of ODE numerical integration.
 *********************************************************************/
enum class Methods {
    RK32,
    RKF45,      ///< Explicit Runge-Kutta-Fehlberf 4(5)
    DOPRI54     ///< Explicit Dormand-Prince 5(4)
};

/********************************************************************
 * @brief Solves an initial value problem for a first-order ODE.
 * 
 * This method provides the numerical integration of a ordinary 
 * differential equation with prescribed initial data:
 * 
 *     dy / dt = rhs(t, y),
 *     y(t0) = y0.
 * 
 * For the numerical integration provided the following methods:
 * 
 * - *DOPRI54* (default): Explicit Dormand-Prince method of order 5(4), [1].
 * - *RKF45*: Explicit Runge-Kutta-Fehlberg method of order 4(5), [2].
 * 
 * References:
 * 
 * - [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
 *       formulae", Journal of Computational and Applied Mathematics
 * - [2] Fehlberg E. "Some experimental results concerning the error
 *       propagation in Runge-Kutta type integration formulas",
 *       National Aeronautics and Space Administration, 1970
 * 
 * 
 * @tparam T Floating point type.
 * @tparam method Which method to use. Default is JustODE::Methods::DOPRI54.
 * @tparam Callable Callable type (the invoke operation is applicable).
 * @tparam Args Types of additional arguments to pass to the RHS.
 * 
 * @param[in] rhs Right-hand side of ODE.
 * @param[in] interval The interval for solving the problem [t0, t_final].
 * @param[in] y0 Initial data y(t0).
 * @param[in] atol (optional) Absolute tolerance. Default is 1e-6.
 * @param[in] rtol (optional) Relative tolerance. Default is 1e-3.
 * @param[in] hmax (optional) Max step size. Default is numerical infinity.
 * @param[in] h_start (optional) Start step size. If not specified or 
 *                    std::nullopt then initial step selected automatic.
 * @return ODEResult<T> object
 *********************************************************************/
template<class T, Methods method = Methods::DOPRI54, class Callable,
         detail::IsFloatingPoint<T> = true>
ODEResult<T> SolveIVP(
    Callable&& rhs,
    const std::array<T, 2>& interval,
    const T& y0,
    std::optional<T> atol    = std::nullopt,
    std::optional<T> rtol    = std::nullopt,
    std::optional<T> hmax    = std::nullopt,
    std::optional<T> h_start = std::nullopt
) {
    if constexpr (method == Methods::RK32) {
        auto solver = RK32<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::RKF45) {
        auto solver = RKF45<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else if constexpr (method == Methods::DOPRI54) {
        auto solver = DOPRI54<T>(atol, rtol, hmax, h_start);
        return solver.Solve(rhs, interval, y0);
    } else {
        static_assert(
            detail::always_false<T>::value,
            "Unknown method. See avaliable methods in JustODE::Methods"
        );
    }
}

}