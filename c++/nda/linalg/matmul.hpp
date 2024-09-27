// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides matrix-matrix an matrix-vector multiplication.
 */

#pragma once

#include "../basic_functions.hpp"
#include "../blas/gemm.hpp"
#include "../blas/gemv.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  namespace detail {

    // Helper variable template to check if the three matrix types can be passed to gemm.
    // The following combinations are allowed (gemm can only be called with 'N', 'T' or 'C' op tags):
    // - C in Fortran layout:
    // -- A/B is not a conj expression and has Fortran layout
    // -- A/B is a conj expression and has C layout
    // - C in C layout:
    // -- A/B is not a conj expression and has C layout
    // -- A/B is a conj expression and has Fortran layout
    template <Matrix A, Matrix B, MemoryMatrix C, bool conj_A = blas::is_conj_array_expr<A>, bool conj_B = blas::is_conj_array_expr<B>>
      requires((MemoryMatrix<A> or conj_A) and (MemoryMatrix<B> or conj_B))
    static constexpr bool is_valid_gemm_triple = []() {
      using blas::has_F_layout;
      if constexpr (has_F_layout<C>) {
        return !(conj_A and has_F_layout<A>)and!(conj_B and has_F_layout<B>);
      } else {
        return !(conj_B and !has_F_layout<B>)and!(conj_A and !has_F_layout<A>);
      }
    }();

    // Get the layout policy for a given array type.
    template <Array A>
    using get_layout_policy = typename std::remove_reference_t<decltype(make_regular(std::declval<A>()))>::layout_policy_t;

  } // namespace detail

  /**
   * @brief Perform a matrix-matrix multiplication.
   *
   * @details It is generic in the sense that it allows the input matrices to belong to a different
   * nda::mem::AddressSpace (as long as they are compatible).
   *
   * If possible, it uses nda::blas::gemm, otherwise it calls nda::blas::gemm_generic.
   *
   * @tparam A nda::Matrix type of lhs operand.
   * @tparam B nda::Matrix type of rhs operand.
   * @param a Left hand side matrix operand.
   * @param b Right hand side matrix operand.
   * @return Result of the matrix-matrix multiplication.
   */
  template <Matrix A, Matrix B>
  auto matmul(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    // check dimensions
    EXPECTS_WITH_MESSAGE(a.shape()[1] == b.shape()[0], "Error in nda::matmul: Dimension mismatch in matrix-matrix product");

    // check address space compatibility
    static constexpr auto L_adr_spc = mem::get_addr_space<A>;
    static constexpr auto R_adr_spc = mem::get_addr_space<B>;
    mem::check_adr_sp_valid<L_adr_spc, R_adr_spc>();

    // get resulting value type, layout policy and matrix type
    using value_t = decltype(get_value_t<A>{} * get_value_t<B>{});
    using layout_policy =
       std::conditional_t<get_layout_info<A>.stride_order == get_layout_info<B>.stride_order, detail::get_layout_policy<A>, C_layout>;
    using matrix_t = basic_array<value_t, 2, layout_policy, 'M', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;

    // perform matrix-matrix multiplication
    auto result = matrix_t(a.shape()[0], b.shape()[1]);
    if constexpr (is_blas_lapack_v<value_t>) {
      // for double or complex value types we use blas::gemm
      // lambda to form a new matrix with the correct value type if necessary
      auto as_container = []<Matrix M>(M &&m) -> decltype(auto) {
        if constexpr (std::is_same_v<get_value_t<M>, value_t> and (MemoryMatrix<M> or blas::is_conj_array_expr<M>))
          return std::forward<M>(m);
        else
          return matrix_t{std::forward<M>(m)};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence this is not necessary.
      // Of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      // check if we can call gemm directly
      if constexpr (detail::is_valid_gemm_triple<decltype(as_container(a)), decltype(as_container(b)), matrix_t>) {
        blas::gemm(1, as_container(a), as_container(b), 0, result);
      } else {
        // otherwise, turn the lhs and rhs first into regular matrices and then call gemm
        blas::gemm(1, make_regular(as_container(a)), make_regular(as_container(b)), 0, result);
      }

    } else {
      // for other value types we use a generic implementation
      blas::gemm_generic(1, a, b, 0, result);
    }
    return result;
  }

  /**
   * @brief Perform a matrix-vector multiplication.
   *
   * @details It is generic in the sense that it allows the input matrix and vector to belong to a different
   * nda::mem::AddressSpace (as long as they are compatible).
   *
   * If possible, it uses nda::blas::gemv, otherwise it calls nda::blas::gemv_generic.
   *
   * @tparam A nda::Matrix type of lhs operand.
   * @tparam X nda::Vector type of rhs operand.
   * @param a Left hand side matrix operand.
   * @param x Right hand side vector operand.
   * @return Result of the matrix-vector multiplication.
   */
  template <Matrix A, Vector X>
  auto matvecmul(A &&a, X &&x) { // NOLINT (temporary views are allowed here)
    // check dimensions
    EXPECTS_WITH_MESSAGE(a.shape()[1] == x.shape()[0], "Error in nda::matvecmul: Dimension mismatch in matrix-vector product");

    // check address space compatibility
    static constexpr auto L_adr_spc = mem::get_addr_space<A>;
    static constexpr auto R_adr_spc = mem::get_addr_space<X>;
    static_assert(L_adr_spc == R_adr_spc, "Error in nda::matvecmul: Matrix-vector product requires arguments with same address spaces");
    static_assert(L_adr_spc != mem::None);

    // get resulting value type and vector type
    using value_t  = decltype(get_value_t<A>{} * get_value_t<X>{});
    using vector_t = vector<value_t, heap<L_adr_spc>>;

    // perform matrix-matrix multiplication
    auto result = vector_t(a.shape()[0]);
    if constexpr (is_blas_lapack_v<value_t>) {
      // for double or complex value types we use blas::gemv
      // lambda to form a new array with the correct value type if necessary
      auto as_container = []<Array B>(B &&b) -> decltype(auto) {
        if constexpr (std::is_same_v<get_value_t<B>, value_t> and (MemoryMatrix<B> or (Matrix<B> and blas::is_conj_array_expr<B>)))
          return std::forward<B>(b);
        else
          return basic_array<value_t, get_rank<B>, C_layout, 'A', heap<L_adr_spc>>{std::forward<B>(b)};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence this is not necessary.
      // Of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      // for expressions of the kind 'conj(M) * V' with a Matrix in Fortran Layout, we have to explicitly
      // form the conj operation in memory as gemv only provides op tags 'N', 'T' and 'C' (hermitian conjugate)
      if constexpr (blas::is_conj_array_expr<decltype(as_container(a))> and blas::has_F_layout<decltype(as_container(a))>) {
        blas::gemv(1, make_regular(as_container(a)), as_container(x), 0, result);
      } else {
        blas::gemv(1, as_container(a), as_container(x), 0, result);
      }
    } else {
      // for other value types we use a generic implementation
      blas::gemv_generic(1, a, x, 0, result);
    }
    return result;
  }

  /** @} */

} // namespace nda
