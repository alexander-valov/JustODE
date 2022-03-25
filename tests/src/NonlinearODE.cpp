#include "doctest.h"
#include "JustODE/JustODE.hpp"

TEST_SUITE("NonlinearODE") {

TEST_CASE("NonlinearODE.logistic_differential_equation") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return std::array<double, 1>{y[0] * (1.0 - y[0])};
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 10};

    // Exact solution
    auto exact_sol = [](const double& t, const double& u0) {
        return u0 * std::exp(t) / (1.0 - u0 + u0 * std::exp(t));
    };

    // Comparison function
    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<double>& y, const double& u0
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i], u0) == doctest::Approx(y[i]));
        }
    };

    // Test
    double y0_max = 10.0;
    int n_y0 = 100;
    for (int i = 0; i < n_y0; i++) {
        std::array<double, 1> y0{i * y0_max / (n_y0 - 1)};

        // RK32
        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(rhs, interval, y0, atol, rtol);
        compare_solution(sol32.t, sol32.y[0], y0[0]);

        // RKF45
        auto sol45 = JustODE::SolveIVP<JustODE::Methods::RKF45>(rhs, interval, y0, atol, rtol);
        compare_solution(sol45.t, sol45.y[0], y0[0]);

        // DOPRI54
        auto sol54 = JustODE::SolveIVP<JustODE::Methods::DOPRI54>(rhs, interval, y0, atol, rtol);
        compare_solution(sol54.t, sol54.y[0], y0[0]);
    }
}

TEST_CASE("NonlinearODE.Riccati_equation") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return std::array<double, 1>{2 * y[0] / t - t * t * y[0] * y[0]};
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{1, 10};
    std::array<double, 1> y0{1};

    // Exact solution
    auto exact_sol = [&](const double& t) {
        return 5 * t * t / (4.0 + t * t * t * t * t);
    };

    // Comparison function
    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<double>& y
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i]) == doctest::Approx(y[i]));
        }
    };

    // Test
    {
        // RK32
        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(rhs, interval, y0, atol, rtol);
        compare_solution(sol32.t, sol32.y[0]);

        // RKF45
        auto sol45 = JustODE::SolveIVP<JustODE::Methods::RKF45>(rhs, interval, y0, atol, rtol);
        compare_solution(sol45.t, sol45.y[0]);

        // DOPRI54
        auto sol54 = JustODE::SolveIVP<JustODE::Methods::DOPRI54>(rhs, interval, y0, atol, rtol);
        compare_solution(sol54.t, sol54.y[0]);
    }
}

TEST_CASE("NonlinearODE.Bernoulli_equation") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return std::array<double, 1>{-y[0] / t + t * y[0] * y[0] * std::log(t)};
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{1, 10};
    std::array<double, 1> y0{-1};

    // Exact solution
    auto exact_sol = [&](const double& t) {
        return -1.0 / (t * (2.0 - t + t * std::log(t)));
    };

    // Comparison function
    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<double>& y
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i]) == doctest::Approx(y[i]));
        }
    };

    // Test
    {
        // RK32
        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(rhs, interval, y0, atol, rtol);
        compare_solution(sol32.t, sol32.y[0]);

        // RKF45
        auto sol45 = JustODE::SolveIVP<JustODE::Methods::RKF45>(rhs, interval, y0, atol, rtol);
        compare_solution(sol45.t, sol45.y[0]);

        // DOPRI54
        auto sol54 = JustODE::SolveIVP<JustODE::Methods::DOPRI54>(rhs, interval, y0, atol, rtol);
        compare_solution(sol54.t, sol54.y[0]);
    }
}

TEST_CASE("NonlinearODE.2x2_homogeneous_system") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            y[0] * (y[0] * y[0] + y[1] * y[1]),
            y[1] * (y[0] * y[0] + y[1] * y[1])
        };
    };
    double atol = 1e-9;
    double rtol = 1e-9;
    std::array<double, 2> interval{0, 49};
    std::array<double, 2> y0{0.1, 0.01};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            5.0 / std::sqrt(2500.0 - (101.0 / 2.0) * t),
            1.0 / std::sqrt(10000.0 - 202.0 * t)
        };
    };

    // Comparison function
    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<std::vector<double>>& y
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i])[0] == doctest::Approx(y[0][i]).epsilon(0.00001));
            CHECK(exact_sol(t[i])[1] == doctest::Approx(y[1][i]).epsilon(0.00001));
        }
    };

    // Test
    {
        // RK32
        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(rhs, interval, y0, atol, rtol);
        compare_solution(sol32.t, sol32.y);

        // RKF45
        auto sol45 = JustODE::SolveIVP<JustODE::Methods::RKF45>(rhs, interval, y0, atol, rtol);
        compare_solution(sol45.t, sol45.y);

        // DOPRI54
        auto sol54 = JustODE::SolveIVP<JustODE::Methods::DOPRI54>(rhs, interval, y0, atol, rtol);
        compare_solution(sol54.t, sol54.y);
    }
}

}