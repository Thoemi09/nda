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
// Authors: Jason Kaye, Miguel Morales, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a C++ interface for various LAPACK routines.
 */

#pragma once

#include "../../blas/tools.hpp"

#include <complex>

#if defined(NDA_HAVE_CUDA)
#include "./cusolver_interface.hpp"
#endif

namespace nda::lapack::f77 {

  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void geqp3(int M, int N, double *A, int LDA, int *JPVT, double *TAU, double *WORK, int LWORK, double *RWORK, int &INFO);
  void geqp3(int M, int N, std::complex<double> *A, int LDA, int *JPVT, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK,
             double *RWORK, int &INFO);

  void orgqr(int M, int N, int K, double *A, int LDA, double *TAU, double *WORK, int LWORK, int &INFO);

  void ungqr(int M, int N, int K, std::complex<double> *A, int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int &INFO);

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info);

  void getri(int N, double *A, int LDA, int const *ipiv, double *work, int lwork, int &info);
  void getri(int N, std::complex<double> *A, int LDA, int const *ipiv, std::complex<double> *work, int lwork, int &info);

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info);
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info);

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info);

  void syev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int &lwork, int &info);

  void heev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int &lwork, double *work2,
            int &info);

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info);

} // namespace nda::lapack::f77

// Useful routines from the BLAS interface
namespace nda::lapack {

  // See nda::blas::get_ld.
  using blas::get_ld;

  // See nda::blas::get_ncols.
  using blas::get_ncols;

  // See nda::blas::get_op.
  using blas::get_op;

  // See nda::blas::has_C_layout.
  using blas::has_C_layout;

  // See nda::blas::has_F_layout.
  using blas::has_F_layout;

  // See nda::blas::is_conj_array_expr.
  using blas::is_conj_array_expr;

} // namespace nda::lapack
