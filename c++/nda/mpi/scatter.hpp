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
 * @brief Provides an MPI scatter function for nda::Array types.
 */

#pragma once

#include "./utils.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

#include <mpi.h>
#include <mpi/mpi.hpp>

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>

/**
 * @ingroup av_mpi
 * @brief Specialization of the `mpi::lazy` class for nda::Array types and the `mpi::tag::scatter` tag.
 *
 * @details An object of this class is returned when scattering nda::Array objects across multiple MPI processes.
 *
 * It models an nda::ArrayInitializer, that means it can be used to initialize and assign to nda::basic_array and
 * nda::basic_array_view objects. The input array/view on the root process will be chunked along the first dimension
 * into equal parts using `mpi::chunk_length` and scattered across all processes in the communicator.
 *
 * See nda::mpi_scatter for an example and more information.
 *
 * @tparam A nda::Array type to be scattered.
 */
template <nda::Array A>
struct mpi::lazy<mpi::tag::scatter, A> {
  /// Value type of the array/view.
  using value_type = typename std::decay_t<A>::value_type;

  /// Type of the array/view stored in the lazy object.
  using stored_type = A;

  /// Array/View to be scattered.
  stored_type rhs;

  /// MPI communicator.
  mpi::communicator comm;

  /// MPI root process.
  const int root{0}; // NOLINT (const is fine here)

  /// Should all processes receive the result. (doesn't make sense for scatter)
  const bool all{false}; // NOLINT (const is fine here)

  /// Size of the array/view to be scattered.
  mutable long scatter_size{0};

  /**
   * @brief Compute the shape of the nda::ArrayInitializer object.
   *
   * @details The input array/view on the root process is chunked along the first dimension into equal (as much as
   * possible) parts using `mpi::chunk_length`.
   *
   * If the extent of the input array along the first dimension is not divisible by the number of processes, processes
   * with lower ranks will receive more data than processes with higher ranks.
   *
   * @warning This makes an MPI call.
   *
   * @return Shape of the nda::ArrayInitializer object.
   */
  [[nodiscard]] auto shape() const {
    auto dims = rhs.shape();
    mpi::broadcast(dims, comm, root);
    scatter_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
    dims[0]      = mpi::chunk_length(dims[0], comm.size(), comm.rank());
    return dims;
  }

  /**
   * @brief Execute the lazy MPI operation and write the result to a target array/view.
   *
   * @details The data will be scattered directly into the memory handle of the target array/view.
   *
   * Throws an exception, if the target array/view is not contiguous with positive strides or if a target view does not
   * have the correct shape.
   *
   * @tparam T nda::Array type with C-layout.
   * @param target Target array/view.
   */
  template <nda::Array T>
    requires(std::decay_t<T>::is_stride_order_C())
  void invoke(T &&target) const { // NOLINT (temporary views are allowed here)
    using namespace nda::detail;

    // special case for non-mpi runs
    if (not mpi::has_env) {
      target = rhs;
      return;
    }

    // check if the target array/view can be used in the MPI call
    check_layout_mpi_compatible(target, "mpi_scatter");

    // get target shape and resize or check the target array/view
    auto dims = shape();
    resize_or_check_if_view(target, dims);

    // scatter the data
    auto target_span = std::span{target.data(), static_cast<std::size_t>(target.size())};
    auto rhs_span    = std::span{rhs.data(), static_cast<std::size_t>(rhs.size())};
    mpi::scatter_range(rhs_span, target_span, scatter_size, comm, root, rhs.indexmap().strides()[0]);
  }
};

namespace nda {

  /**
   * @ingroup av_mpi
   * @brief Implementation of an MPI scatter for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function scatters a C-ordered input array/view from a root process across all processes in the given
   * communicator. The array/view is chunked into equal parts along the first dimension using `mpi::chunk_length`.
   *
   * Throws an exception, if the given array/view on the root process is not contiguous with positive strides.
   * Furthermore, it is expected that the input arrays/views have the same rank and a C-layout on all processes.
   *
   * This function is lazy, i.e. it returns an mpi::lazy<mpi::tag::scatter, A> object without performing the actual MPI
   * operation. Since the returned object models an nda::ArrayInitializer, it can be used to initialize/assign to
   * nda::basic_array and nda::basic_array_view objects:
   *
   * @code{.cpp}
   * // create an array on all processes
   * nda::array<int, 2> A(10, 4);
   *
   * // ...
   * // fill array on root process
   * // ...
   *
   * // scatter the array to all processes
   * nda::array<int, 2> B = mpi::scatter(A);
   * @endcode
   *
   * Here, the array `B` has the shape `(10 / comm.size(), 4)` on each process (assuming that 10 is a multiple of
   * `comm.size()`).
   *
   * @warning MPI calls are done in the `invoke` and `shape` methods of the `mpi::lazy` object. If one rank calls one of
   * these methods, all ranks in the communicator need to call the same method. Otherwise, the program will deadlock.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array or view to be scattered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the scatter (not used).
   * @return An mpi::lazy<mpi::tag::scatter, A> object modelling an nda::ArrayInitializer.
   */
  template <typename A>
  ArrayInitializer<std::remove_reference_t<A>> auto mpi_scatter(A &&a, mpi::communicator comm = {}, int root = 0, bool all = false)
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  {
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_ranks(a, comm), "Ranks of arrays/views must be equal in nda::mpi_scatter")
    if (comm.rank() == root) {
      detail::check_layout_mpi_compatible(a, "mpi_scatter");
    }
    return mpi::lazy<mpi::tag::scatter, A>{std::forward<A>(a), comm, root, all};
  }

} // namespace nda
