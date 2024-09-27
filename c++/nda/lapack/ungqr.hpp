// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Jason Kaye

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `ungqr` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `ungqr` routine.
   *
   * @details Generates an m-by-n complex matrix \f$ \mathbf{Q} \f$ with orthonormal columns, which is defined as the
   * first n columns of a product of k elementary reflectors of order m
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * as returned by `geqrf`.
   *
   * @tparam A nda::MemoryMatrix with complex value type.
   * @tparam TAU nda::MemoryVector with complex value type.
   * @param a Input/output matrix. On entry, the i-th column must contain the vector which defines the elementary
   * reflector \f$ H(i) \; , i = 1,2,...,K \f$, as returned by `geqrf` in the first k columns. On exit, the m-by-n
   * matrix \f$ \mathbf{Q} \f$.
   * @param tau Input vector. `tau(i)` must contain the scalar factor of the elementary reflector \f$ \mathbf{H}(i) \f$,
   * as returned by `geqrf`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector TAU>
    requires(mem::on_host<A> and std::is_same_v<std::complex<double>, get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, TAU>)
  int ungqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::ungqr: C order is not supported");
    static_assert(mem::have_host_compatible_addr_space<A, TAU>, "Error in nda::lapack::ungqr: Only CPU is supported");

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    auto [m, n] = a.shape();
    auto k      = tau.size();
    int info    = 0;
    lapack::f77::ungqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), &bufferSize_T, -1, info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work(bufferSize);
    lapack::f77::ungqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), work.data(), bufferSize, info);

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::ungqr: info = " << info;
    return info;
  }

} // namespace nda::lapack
