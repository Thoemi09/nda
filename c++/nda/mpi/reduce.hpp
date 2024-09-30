// Copyright (c) 2020-2024 Simons Foundation
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
 * @brief Provides an MPI reduce function for nda::Array types.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../exceptions.hpp"
#include "../macros.hpp"
#include "../map.hpp"
#include "../traits.hpp"

#include <mpi.h>
#include <mpi/mpi.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

/**
 * @ingroup av_mpi
 * @brief Specialization of the `mpi::lazy` class for nda::Array types and the `mpi::tag::reduce` tag.
 *
 * @details An object of this class is returned when reducing nda::Array objects across multiple MPI processes.
 *
 * It models an nda::ArrayInitializer, that means it can be used to initialize and assign to nda::basic_array and
 * nda::basic_array_view objects. The result will be the reduction of the input arrays/views with respect to the given
 * MPI operation.
 *
 * See nda::mpi_reduce for an example.
 *
 * @tparam A nda::Array type to be reduced.
 */
template <nda::Array A>
struct mpi::lazy<mpi::tag::reduce, A> {
  /// Value type of the array/view.
  using value_type = typename std::decay_t<A>::value_type;

  /// Type of the array/view stored in the lazy object.
  using stored_type = A;

  /// Array/View to be reduced.
  stored_type rhs;

  /// MPI communicator.
  mpi::communicator comm;

  /// MPI root process.
  const int root{0}; // NOLINT (const is fine here)

  /// Should all processes receive the result.
  const bool all{false}; // NOLINT (const is fine here)

  /// MPI reduction operation.
  const MPI_Op op{MPI_SUM}; // NOLINT (const is fine here)

  /**
   * @brief Compute the shape of the nda::ArrayInitializer object.
   *
   * @details The shape of the initializer objects depends on the MPI rank and whether it receives the data or not:
   *
   * - On receiving ranks, the shape is the same as the shape of the input array/view.
   * - On non-receiving ranks, the shape has the rank of the input array/view with only zeros, i.e. it is empty.
   *
   * @return Shape of the nda::ArrayInitializer object.
   */
  [[nodiscard]] auto shape() const {
    if ((comm.rank() == root) || all) return rhs.shape();
    return std::array<long, std::remove_cvref_t<stored_type>::rank>{};
  }

  /**
   * @brief Execute the lazy MPI operation and write the result to a target array/view.
   *
   * @details The data will be reduced directly into the memory handle of the target array/view. If the target
   * array/view is the same as the input array/view, i.e. if their data pointers are the same, the reduction is
   * performed in-place.
   *
   * Types which cannot be reduced directly, i.e. which do not have an MPI type, are reduced element-wise.
   *
   * For MPI compatible types, the function throws an exception, if
   * - the input array/view is not contiguous with positive strides.
   * - the target array/view is not contiguous with positive strides on receiving ranks.
   * - the operation is performed in-place but the target and input array have different sizes.
   * - the operation is performed out-of-place and the memory of the target and input array/view overlap.
   *
   * @tparam T nda::Array type of the target array/view.
   * @param target Target array/view.
   */
  template <nda::Array T>
  void invoke(T &&target) const { // NOLINT (temporary views are allowed here)
    // special case for non-mpi runs
    if (not mpi::has_env) {
      target = rhs;
      return;
    }

    // perform the reduction
    if constexpr (not mpi::has_mpi_type<value_type>) {
      // arrays/views of non-MPI types are reduced element-wise
      target = nda::map([this](auto const &x) { return mpi::reduce(x, this->comm, this->root, this->all, this->op); })(rhs);
    } else {
      using namespace nda::detail;
      bool in_place = (target.data() == rhs.data());

      // for MPI-types we have to perform some checks on the input and target arrays/views
      check_layout_mpi_compatible(rhs, "mpi_reduce");
      if ((comm.rank() == root) || all) {
        check_layout_mpi_compatible(target, "mpi_reduce");
        if (in_place) {
          if (rhs.size() != target.size())
            NDA_RUNTIME_ERROR << "Error in MPI reduce for nda::Array: In-place reduction requires both arrays/views to be same size";
        } else {
          nda::resize_or_check_if_view(target, shape());
          if (std::abs(target.data() - rhs.data()) < rhs.size()) NDA_RUNTIME_ERROR << "Error in MPI reduce for nda::Array: Overlapping arrays";
        }
      }

      // reduce the data
      auto target_span = std::span{target.data(), static_cast<std::size_t>(target.size())};
      if (in_place) {
        mpi::reduce_in_place_range(target_span, comm, root, all, op);
      } else {
        auto rhs_span = std::span{rhs.data(), static_cast<std::size_t>(rhs.size())};
        mpi::reduce_range(rhs_span, target_span, comm, root, all, op);
      }
    }
  }
};

namespace nda {

  /**
   * @ingroup av_mpi
   * @brief Implementation of an MPI reduce for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function reduces input arrays/views from all processes in the given communicator and makes the result
   * available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It is expected that all arrays/views have the same shape on all processes.
   *
   * This function is lazy, i.e. it returns an mpi::lazy<mpi::tag::reduce, A> object without performing the actual MPI
   * operation. Since the returned object models an nda::ArrayInitializer, it can be used to initialize/assign to
   * nda::basic_array and nda::basic_array_view objects:
   *
   * @code{.cpp}
   * // create an array on all processes
   * nda::array<int, 2> A(3, 4);
   *
   * // ...
   * // fill array on each process
   * // ...
   *
   * // reduce the array on the root process
   * nda::array<int, 2> B = mpi::reduce(A);
   * @endcode
   *
   * Here, the array `B` has the shape `(3, 4)` on the root process and `(0, 0)` on all other processes.
   *
   * @warning MPI calls are done in the `invoke` method of the `mpi::lazy` object. If one rank calls this methods, all
   * ranks in the communicator need to call the same method. Otherwise, the program will deadlock.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array or view to be reduced.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return An mpi::lazy<mpi::tag::reduce, A> object modelling an nda::ArrayInitializer.
   */
  template <typename A>
  ArrayInitializer<std::remove_reference_t<A>> auto mpi_reduce(A &&a, mpi::communicator comm = {}, int root = 0, bool all = false,
                                                               MPI_Op op = MPI_SUM)
    requires(is_regular_or_view_v<A>)
  {
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_shapes(a, comm), "Shapes of arrays/views must be equal in nda::mpi_reduce");
    return mpi::lazy<mpi::tag::reduce, A>{std::forward<A>(a), comm, root, all, op};
  }

} // namespace nda
