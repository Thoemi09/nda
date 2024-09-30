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
 * @brief Provides an MPI gather function for nda::Array types.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../layout/range.hpp"
#include "../macros.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#include <mpi/mpi.hpp>

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>

/**
 * @ingroup av_mpi
 * @brief Specialization of the `mpi::lazy` class for nda::Array types and the `mpi::tag::gather` tag.
 *
 * @details An object of this class is returned when gathering nda::Array objects across multiple MPI processes.
 *
 * It models an nda::ArrayInitializer, that means it can be used to initialize and assign to nda::basic_array and
 * nda::basic_array_view objects. The result will be a concatenation of the input arrays/views along their first
 * dimension.
 *
 * See nda::mpi_gather for an example and more information.
 *
 * @tparam A nda::Array type to be gathered.
 */
template <nda::Array A>
struct mpi::lazy<mpi::tag::gather, A> {
  /// Value type of the array/view.
  using value_type = typename std::decay_t<A>::value_type;

  /// Type of the array/view stored in the lazy object.
  using stored_type = A;

  /// Array/View to be gathered.
  stored_type rhs;

  /// MPI communicator.
  mpi::communicator comm;

  /// MPI root process.
  const int root{0}; // NOLINT (const is fine here)

  /// Should all processes receive the result.
  const bool all{false}; // NOLINT (const is fine here)

  /// Size of the gathered array/view.
  mutable long gathered_size{0};

  /**
   * @brief Compute the shape of the nda::ArrayInitializer object.
   *
   * @details The input arrays/views are simply concatenated along their first dimension. The shape of the initializer
   * object depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, the shape is the same as the shape of the input array/view except for the first dimension,
   * which is the sum of the extents of all input arrays/views along the first dimension.
   * - On non-receiving ranks, the shape is empty, i.e. `(0,0,...,0)`.
   *
   * @warning This makes an MPI call.
   *
   * @return Shape of the nda::ArrayInitializer object.
   */
  [[nodiscard]] auto shape() const {
    auto dims     = rhs.shape();
    dims[0]       = mpi::all_reduce(dims[0], comm);
    gathered_size = std::accumulate(dims.begin(), dims.end(), 1l, std::multiplies<>());
    if (!all && comm.rank() != root) dims = nda::stdutil::make_initialized_array<dims.size()>(0l);
    return dims;
  }

  /**
   * @brief Execute the lazy MPI operation and write the result to a target array/view.
   *
   * @details The data will be gathered directly into the memory handle of the target array/view.
   *
   * Throws an exception if
   * - the target array/view is not contiguous with positive strides on receiving ranks,
   * - a target view does not have the correct shape on receiving ranks,
   * - the target array/view is not in C-layout on receiving ranks or
   * - one of the MPI calls fails.
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

    // get target shape, resize or check the target array/view and prepare output span
    auto dims        = shape();
    auto target_span = std::span{target.data(), 0};
    if (all || (comm.rank() == root)) {
      // check if the target array/view can be used in the MPI call
      check_layout_mpi_compatible(target, "mpi_gather");

      // resize/check the size of the target array/view
      nda::resize_or_check_if_view(target, dims);

      // prepare the output span
      target_span = std::span{target.data(), static_cast<std::size_t>(target.size())};
    }

    // gather the data
    auto rhs_span = std::span{rhs.data(), static_cast<std::size_t>(rhs.size())};
    mpi::gather_range(rhs_span, target_span, gathered_size, comm, root, all);
  }
};

namespace nda {

  /**
   * @ingroup av_mpi
   * @brief Implementation of an MPI gather for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function gathers C-ordered input arrays/views from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`). The
   * arrays/views are joined along the first dimension.
   *
   * Throws an exception, if a given array/view is not contiguous with positive strides. Furthermore, it is expected
   * that the input arrays/views have the same shape on all processes except for the first dimension.
   *
   * This function is lazy, i.e. it returns an mpi::lazy<mpi::tag::gather, A> object without performing the actual MPI
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
   * // gather the arrays on the root process
   * nda::array<int, 2> B = mpi::gather(A);
   * @endcode
   *
   * Here, the array `B` has the shape `(3 * comm.size(), 4)` on the root process and `(0, 0)` on all other processes.
   *
   * @warning MPI calls are done in the `invoke` and `shape` methods of the `mpi::lazy` object. If one rank calls one of
   * these methods, all ranks in the communicator need to call the same method. Otherwise, the program will deadlock.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type with C-layout.
   * @param a Array or view to be gathered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return An mpi::lazy<mpi::tag::gather, A> object modelling an nda::ArrayInitializer.
   */
  template <typename A>
  ArrayInitializer<std::remove_reference_t<A>> auto mpi_gather(A &&a, mpi::communicator comm = {}, int root = 0, bool all = false)
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  {
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_shapes(a(nda::range(1), nda::ellipsis{}), comm),
                         "Shapes of arrays/views must be equal save the first one in nda::mpi_gather");
    detail::check_layout_mpi_compatible(a, "mpi_gather");
    return mpi::lazy<mpi::tag::gather, A>{std::forward<A>(a), comm, root, all};
  }

} // namespace nda
