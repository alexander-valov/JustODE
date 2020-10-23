#include <iostream>

#include "RungeKutta.hpp"

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

    int nfev = 0;
    auto rhs = [&](double t, double y) {
        nfev += 1;
        return -2 * y + std::exp(-2 * (t - 6) * (t - 6));
    };

    nfev = 0;
    auto solver = RungeKutta45(0.05, 2.0, 1.0e-6, 1.0e-10);
    auto [flag, tvals, yvals] = solver.Solve(rhs, {0.0, 15.0}, 1.0);
    std::cout << "nfev: " << nfev << "\n";
    std::cout << "size: " << tvals.size() << "\n";
    PrintResults(flag, tvals, yvals);
}
