#include <iostream>

#include "JustODE/JustODE.hpp"
#include "matplot/matplot.h"

template <class T>
void PrintResults(int flag, const std::vector<T>& tvals, const std::vector<T>& yvals) {
    std::cout << "flag: " << flag << "\n";
    std::cout << "t = [";
    for (size_t i = 0; i < tvals.size(); i++) {
        std::cout << tvals[i];
        if (i < tvals.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "y = [";
    for (size_t i = 0; i < yvals.size(); i++) {
        std::cout << yvals[i];
        if (i < yvals.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main() {

    auto rhs = [&](const double& t, const double& y) {
        return -2 * y + std::exp(-2 * (t - 6) * (t - 6));
    };

    std::array<double, 2> interval{0, 15};
    double y0 = 1;
    double atol = 1e-6;
    double rtol = 1e-3;


    std::cout << "\n\nRunge-Kutta 3(2):\n";
    auto [tvalsRK32, yvalsRK32, flagRK32, messageRK32, nfevRK32] = JustODE::SolveIVP<double, JustODE::Methods::RK32>(
        rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
    );
    std::cout << "nfev: " << nfevRK32 << "\n";
    std::cout << "size: " << tvalsRK32.size() << "\n";
    PrintResults(flagRK32, tvalsRK32, yvalsRK32);


    std::cout << "\n\nRunge-Kutta-Fehlberg 4(5):\n";
    auto [tvalsRKF45, yvalsRKF45, flagRKF45, messageRKF45, nfevRKF45] = JustODE::SolveIVP<double, JustODE::Methods::RKF45>(
        rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
    );
    std::cout << "nfev: " << nfevRKF45 << "\n";
    std::cout << "size: " << tvalsRKF45.size() << "\n";
    PrintResults(flagRKF45, tvalsRKF45, yvalsRKF45);


    std::cout << "\n\nDormand-Prince 5(4):\n";
    auto [tvalsDOPR, yvalsDOPR, flagDOPR, messageDOPR, nfevDOPR] = JustODE::SolveIVP<double, JustODE::Methods::DOPRI54>(
        rhs, interval, y0, atol, rtol, std::nullopt, std::nullopt
    );
    std::cout << "nfev: " << nfevDOPR << "\n";
    std::cout << "size: " << tvalsDOPR.size() << "\n";
    PrintResults(flagDOPR, tvalsDOPR, yvalsDOPR);


    auto exact_sol = [](const double& t) {
        return 0.25 * std::exp(-2 * t) * (4 + std::sqrt(2 * M_PI) * std::exp(25.0 / 2.0) * (std::erf(13.0 / std::sqrt(2)) + std::erf((-13.0 + 2 * t) / std::sqrt(2))));
    };
    std::vector<double> exact_x = matplot::linspace(interval[0], interval[1], 300);
    std::vector<double> exact_y = matplot::transform(exact_x, exact_sol);

    auto p = matplot::plot(exact_x, exact_y, tvalsRK32, yvalsRK32, tvalsRKF45, yvalsRKF45, tvalsDOPR, yvalsDOPR);
    p[0]->line_width(5);
    p[1]->line_width(2)
         .line_style("--")
         .marker(matplot::line_spec::marker_style::circle)
         .marker_size(10);
    p[2]->line_width(2)
         .line_style("--")
         .marker(matplot::line_spec::marker_style::cross)
         .marker_size(10);
    p[3]->line_width(2)
         .line_style("--")
         .marker(matplot::line_spec::marker_style::diamond)
         .marker_size(10);

    matplot::show();

    return 0;
}
