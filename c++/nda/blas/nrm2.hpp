// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `nrm2` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::blas {

  /**
   * @ingroup linalg_blas
   * @brief Interface to the BLAS `nrm2` routine.
   *
   * @details Computes the Euclidean norm of a vector. This function calculates
   * \f[
   *   \|\mathbf{x}\|_2 = \sqrt{\sum_i |x_i|^2}
   * \f]
   *
   * @tparam X nda::MemoryVector type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @return The Euclidean norm of the vector.
   */
  template <MemoryVector X>
    requires(is_blas_lapack_v<get_value_t<X>>)
  auto nrm2(X const &x) {
    // perform actual library call
    return f77::nrm2(x.size(), x.data(), x.indexmap().strides()[0]);
  }

} // namespace nda::blas
