#include <nda/nda.hpp>
#include <iostream>

int main() {
    // create two 3D vectors
    auto a = nda::vector<double>{1.0, 2.0, 3.0};
    auto b = nda::vector<double>{-1.0, 2.0, 3.0};

    // compute their cross product
    auto c = nda::linalg::cross_product(a, b);

    // print the result
    std::cout << "c = " << c << std::endl;
}