#include <nda/nda.hpp>
#include <nda/blas.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    // Test 1: Simple double vector - 3-4-5 right triangle
    auto v1 = nda::vector<double>{3.0, 4.0};
    double norm1 = nda::blas::nrm2(v1);
    std::cout << "Test 1: norm of [3.0, 4.0] = " << norm1 << " (expected: 5.0)" << std::endl;
    
    // Verify the result
    double expected1 = 5.0;
    if (std::abs(norm1 - expected1) < 1e-10) {
        std::cout << "  ✓ Test 1 passed!" << std::endl;
    } else {
        std::cout << "  ✗ Test 1 failed!" << std::endl;
        return 1;
    }
    
    // Test 2: Unit vector
    auto v2 = nda::vector<double>{1.0, 0.0, 0.0};
    double norm2 = nda::blas::nrm2(v2);
    std::cout << "\nTest 2: norm of [1.0, 0.0, 0.0] = " << norm2 << " (expected: 1.0)" << std::endl;
    
    double expected2 = 1.0;
    if (std::abs(norm2 - expected2) < 1e-10) {
        std::cout << "  ✓ Test 2 passed!" << std::endl;
    } else {
        std::cout << "  ✗ Test 2 failed!" << std::endl;
        return 1;
    }
    
    // Test 3: Complex double vector
    using nda::dcomplex;
    auto v3 = nda::vector<dcomplex>{dcomplex{1.0, 0.0}, dcomplex{0.0, 1.0}};
    double norm3 = nda::blas::nrm2(v3);
    std::cout << "\nTest 3: norm of [(1+0i), (0+1i)] = " << norm3 << " (expected: " << std::sqrt(2.0) << ")" << std::endl;
    
    double expected3 = std::sqrt(2.0);
    if (std::abs(norm3 - expected3) < 1e-10) {
        std::cout << "  ✓ Test 3 passed!" << std::endl;
    } else {
        std::cout << "  ✗ Test 3 failed!" << std::endl;
        return 1;
    }
    
    // Test 4: Complex double vector with non-trivial values
    auto v4 = nda::vector<dcomplex>{dcomplex{3.0, 4.0}, dcomplex{0.0, 0.0}};
    double norm4 = nda::blas::nrm2(v4);
    std::cout << "\nTest 4: norm of [(3+4i), (0+0i)] = " << norm4 << " (expected: 5.0)" << std::endl;
    
    double expected4 = 5.0;
    if (std::abs(norm4 - expected4) < 1e-10) {
        std::cout << "  ✓ Test 4 passed!" << std::endl;
    } else {
        std::cout << "  ✗ Test 4 failed!" << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ All tests passed!" << std::endl;
    return 0;
}
