// Copyright (c) 2020-2023 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides various utility functions used by the MPI interface of nda.
 */

#pragma once

#include "../concepts.hpp"
#include "../exceptions.hpp"

#include <mpi/mpi.hpp>

#include <string>

namespace nda::detail {

  // Check if the layout of an array/view is contiguous with positive strides, otherwise throw an exception.
  template <nda::Array A>
  void check_layout_mpi_compatible(const A &a, const std::string &func) {
    if (not a.is_contiguous() or not a.has_positive_strides())
      NDA_RUNTIME_ERROR << "Error in function " << func << ": Array is not contiguous with positive strides";
  }

  // Check if the ranks of arrays/views are the same on all processes.
  template <nda::Array A>
  bool have_mpi_equal_ranks(const A &a, const mpi::communicator &comm) {
    return mpi::all_equal(a.rank, comm);
  }

  // Check if the shape of arrays/views are the same on all processes.
  template <nda::Array A>
  bool have_mpi_equal_shapes(const A &a, const mpi::communicator &comm) {
    return have_mpi_equal_ranks(a, comm) && mpi::all_equal(a.shape(), comm);
  }

} // namespace nda::detail
