#pragma once

#include <type_traits>
#include <string>
#include <cmath>
#include <map>

#include "LUPFactorization.hpp"
#include "TraitsHelper.hpp"
#include "Operations.hpp"

namespace JustODE {

namespace detail {

template<class T, class Container>
struct RootSolution {
    Container solution;   ///< The root of a nonlinear equation
    int status;           ///< The solver termination status
    std::string message;  ///< Termination cause description
    T error;              ///< The algorithm error
    std::size_t niter;    ///< The actual number of iterations
};

template<class T, class Container, class JacobianType>
class LevenbergMarquardt {

public:
    LevenbergMarquardt(const T& tol = T(1.0e-8), std::size_t maxiter = 500, const T& mumin = T(1.0e-6))
      : tolerance(tol), max_iter(maxiter), mu_min(mumin)
    {
        static_assert(
            type_traits::is_real_container<Container>::value,
            "The type `Container` must be floating point and iterable (supports std::begin() and std::end())"
        );
    }

    const T& Tolerance() const { return tolerance; }
    std::size_t MaxIter() const { return max_iter; }
    const T& MuMin() const { return mu_min; }

    void SetTolerance(const T& tol) { tolerance = tol; }
    void SetMaxIter(std::size_t maxiter) { max_iter = maxiter; }
    void SetMuMin(const T& mumin) { mu_min = mumin; }

    template<class CallableF, class CallableJ>
    RootSolution<T, Container> Solve(
        CallableF&& func,
        CallableJ&& jacobian,
        const Container& x0
    ) const {
        static_assert(
            std::is_invocable_r_v<Container, CallableF&&, const Container&>,
            "Invalid signature or return type of the nonlinear system functor!"
        );
        static_assert(
            std::is_invocable_r_v<JacobianType, CallableJ&&, const Container&>,
            "Invalid signature or return type of the Jacobian matrix functor!"
        );

        // -----------------------------------------------------------
        // Step 1
        // -----------------------------------------------------------
        T error = 2 * tolerance + 1;
        Container x_k = x0;
        T mu_k = 10 * mu_min;
        Container F_k = func(x_k);
        JacobianType J_k = jacobian(x_k);
        JacobianType J_k_T = type_traits::transpose(J_k);
        std::size_t iteration = 0;

        while (error > tolerance && iteration < max_iter) {
            T squared_norm_F_k = SquaredNorm(F_k, T(0));

            // -----------------------------------------------------------
            // Step 2
            // -----------------------------------------------------------
            // Calculate LM-Matrix and factorize
            T lambda_k = mu_k * squared_norm_F_k;
            JacobianType matrix = J_k_T * J_k;
            for (int i = 0; i < type_traits::rows(matrix); i++) {
                matrix(i, i) += lambda_k;
            }
            LUPFactorization<JacobianType&> solver(matrix);

            // Calculate RHS for d_k
            Container rhs_d = MultScalar(-1, MVProd(J_k_T, F_k));
            Container d_k = solver.Solve(rhs_d);

            // Calculate RHS for d_k_hat
            Container y_k = PlusCwise(x_k, d_k);
            Container F_y_k = func(y_k);
            Container rhs_d_hat = MultScalar(-1, MVProd(J_k_T, F_y_k));
            Container d_k_hat = solver.Solve(rhs_d_hat);

            // -----------------------------------------------------------
            // Step 3
            // -----------------------------------------------------------
            Container z_k = PlusCwise(y_k, d_k_hat);
            Container F_z_k = func(z_k);
            T Ared = squared_norm_F_k - SquaredNorm(F_z_k, T(0));
            T Pred = squared_norm_F_k         - SquaredNorm(PlusCwise(F_k,   MVProd(J_k, d_k)),     T(0)) +
                     SquaredNorm(F_y_k, T(0)) - SquaredNorm(PlusCwise(F_y_k, MVProd(J_k, d_k_hat)), T(0));
            T r_k = 2 * p2;
            if (Pred > 0) { r_k = Ared / Pred; }
            if (r_k >= p0) { x_k = z_k; }

            // -----------------------------------------------------------
            // Step 4
            // -----------------------------------------------------------
            if (r_k < p1) {
                mu_k = 4 * mu_k;
            } else if (r_k > p2) {
                std::max(T(0.25) * mu_k, mu_min);
            }

            // Calculate error
            F_k = func(x_k);
            J_k = jacobian(x_k);
            J_k_T = type_traits::transpose(J_k);
            error = std::sqrt(SquaredNorm(MVProd(J_k_T, F_k), T(0)));
            ++iteration;
        }

        RootSolution<T, Container> result;
        result.solution = std::move(x_k);
        if (iteration < max_iter) {
            result.status = 0;
        } else {
            result.status = 1;
        }
        result.message = messages_.at(result.status);
        result.error = error;
        result.niter = iteration;
        return result;;
    }

protected:
    T tolerance = T(1.0e-8);
    std::size_t max_iter = 500;
    T mu_min = T(1.0e-6);
    T p0 = T(1.0e-4);
    T p1 = T(0.25);
    T p2 = T(0.75);

    const std::map<int, std::string> messages_{
        {0, "Success"},
        {1, "Terminated. Excceding maximum iterations"}
    }; ///< The array of possible solver messages
};

}

}