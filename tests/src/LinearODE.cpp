#include "doctest.h"
#include "JustODE/JustODE.hpp"

TEST_SUITE("LinearODE") {

TEST_CASE("LinearODE.exponent") {
    // Problem statement
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 100};
    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return y;
    };

    // Exact solution
    auto exact_sol = [](const double& t, const double& y0) {
        return y0 * std::exp(t);
    };

    // Comparison function
    auto compare_solution = [&](
        const std::vector<double>& t, const std::vector<double>& y, const double& y0
    ) {
        for (std::size_t i = 0; i < t.size(); i++) {
            CHECK(exact_sol(t[i], y0) == doctest::Approx(y[i]));
        }
    };

    // Test
    double y0_max = 10.0;
    int n_y0 = 15;
    for (int i = 0; i < n_y0; i++) {
        std::array<double, 1> y0 = {i * y0_max / (n_y0 - 1)};

        // RK32
        auto sol32 = JustODE::SolveIVP<JustODE::Methods::RK32>(
            rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
        );
        compare_solution(sol32.t, sol32.y[0], y0[0]);

        // RKF45
        auto sol45 = JustODE::SolveIVP<JustODE::Methods::RKF45>(
            rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
        );
        compare_solution(sol45.t, sol45.y[0], y0[0]);

        // DOPRI54
        auto sol54 = JustODE::SolveIVP<JustODE::Methods::DOPRI54>(
            rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
        );
        compare_solution(sol54.t, sol54.y[0], y0[0]);
    }
    
}

TEST_CASE("LinearODE.equation_with_variable_coefficients") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 1>& y) {
        return std::array<double, 1>{3 * t - y[0] / t};
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{1, 10};
    double alpha = 101.0;
    std::array<double, 1> y0{alpha};

    // Exact solution
    auto exact_sol = [&](const double& t) {
        return t * t + (alpha - 1) / t;
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

TEST_CASE("LinearODE.second_order_D>0") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            y[1],
            - 3 * y[1] - 2 * y[0] + 1.0 / (1.0 + std::exp(t))
        };
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 20};
    std::array<double, 2> y0{0, 1};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            std::exp(-2 * t) * (1 + std::exp(t)) * std::log(0.5 * (1.0 + std::exp(t))),
            std::exp(-2 * t) * (std::exp(t) + std::log(4.0) - std::exp(t) * std::log(0.5 * (1.0 + std::exp(t))) - 2 * std::log(1.0 + std::exp(t)))
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

TEST_CASE("LinearODE.second_order_D=0") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            y[1],
            -2 * y[1] - y[0] + 3 * std::sin(t) + 2 * std::cos(t)
        };
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 10};
    std::array<double, 2> y0{1, -10};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            0.5 * std::exp(-t) * (5.0 - 17.0 * t)   - (3.0 / 2.0) * std::cos(t) + std::sin(t),
            0.5 * std::exp(-t) * (-22.0 + 17.0 * t) + (3.0 / 2.0) * std::sin(t) + std::cos(t)
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

TEST_CASE("LinearODE.second_order_D<0") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            y[1],
            - y[0] + 3 * std::sin(t) + 2 * std::cos(t)
        };
    };
    double atol = 1e-10;
    double rtol = 1e-10;
    std::array<double, 2> interval{0, 20};
    std::array<double, 2> y0{0, 1};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            (5.0 / 2.0 + t) * std::sin(t) - (3.0 / 2.0) * t * std::cos(t),
            (1.0 + (3.0 / 2.0) * t) * std::sin(t) + (1.0 + t) * std::cos(t)
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

TEST_CASE("LinearODE.2x2_homogeneous_system") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            3 * y[0] - 4 * y[1],
            4 * y[0] - 7 * y[1]
        };
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 20};
    std::array<double, 2> y0{1, 1};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            (2.0 / 3.0) * std::exp(t) + (1.0 / 3.0) * std::exp(-5 * t),
            (1.0 / 3.0) * std::exp(t) + (2.0 / 3.0) * std::exp(-5 * t)
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

TEST_CASE("LinearODE.2x2_inhomogeneous_system") {
    // Problem statement
    auto rhs = [&](const double& t, const std::array<double, 2>& y) {
        return std::array<double, 2>{
            -y[1],
            y[0] + std::cos(t)
        };
    };
    double atol = 1e-8;
    double rtol = 1e-8;
    std::array<double, 2> interval{0, 10};
    std::array<double, 2> y0{-1, 1};

    // Exact solution
    auto exact_sol = [](const double& t) {
        return std::array<double, 2>{
            -std::cos(t) - 0.5 * (2.0 + t) * std::sin(t),
            0.5 * ((2.0 + t) * std::cos(t) - std::sin(t))
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