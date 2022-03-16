#include "doctest.h"
#include "JustODE/JustODE.hpp"

TEST_SUITE("NonStiff.LinearODE") {

TEST_CASE("NonStiff.LinearODE.exponent") {
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 2 * M_PI};

    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return y;
    };

    auto exact_sol = [](const double& t, const double& y0) {
        return y0 * std::exp(t);
    };

    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<double>& y, const double& y0
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i], y0) == doctest::Approx(y[i]));
        }
    };

    double y0_max = 10.0;
    int n_y0 = 15;
    for (int i = 0; i < n_y0; i++) {
        std::array<double, 1> y0 = {i * y0_max / (n_y0 - 1)};

        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(
            rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
        );
        compare_solution(sol32.t, sol32.y[0], y0[0]);
    }
    
}

}