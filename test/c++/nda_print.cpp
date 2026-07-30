// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <complex>
#include <cstdint>
#include <sstream>
#include <string>

// All expected strings below have been validated against numpy 1.26.4, except where nda deliberately diverges
// (no uniform float formatting, hence e.g. "0+1j" instead of numpy's "0.+1.j", and no line wrapping).

// Render via the stream operator.
template <typename A>
std::string via_stream(A const &a) {
  std::ostringstream os;
  os << a;
  return os.str();
}

// Fill an array with 0, 1, 2, ... in row-major order.
template <typename A>
void iota_fill(A &a) {
  long n = 0;
  for (auto &x : a) x = n++;
}

TEST(NDAPrint, Rank1) {
  nda::array<long, 1> A{1, 2, 3};
  EXPECT_EQ(nda::to_string(A), "[1 2 3]");
  EXPECT_EQ(via_stream(A), "[1 2 3]");
}

TEST(NDAPrint, Rank1RightAlignedToCommonWidth) {
  EXPECT_EQ(nda::to_string(nda::array<long, 1>{-1, 10, 3}), "[-1 10  3]");
}

TEST(NDAPrint, Rank2) {
  EXPECT_EQ(nda::to_string(nda::array<long, 2>{{0, 1, 2}, {3, 4, 5}}),
            "[[0 1 2]\n"
            " [3 4 5]]");
}

TEST(NDAPrint, Rank2CommonWidth) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;
  EXPECT_EQ(nda::to_string(A),
            "[[ 0  1  2]\n"
            " [10 11 12]]");
}

TEST(NDAPrint, Rank3) {
  nda::array<long, 3> A(2, 2, 3);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A),
            "[[[ 0  1  2]\n"
            "  [ 3  4  5]]\n"
            "\n"
            " [[ 6  7  8]\n"
            "  [ 9 10 11]]]");
}

TEST(NDAPrint, Rank4) {
  nda::array<long, 4> A(2, 2, 2, 2);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A),
            "[[[[ 0  1]\n"
            "   [ 2  3]]\n"
            "\n"
            "  [[ 4  5]\n"
            "   [ 6  7]]]\n"
            "\n"
            "\n"
            " [[[ 8  9]\n"
            "   [10 11]]\n"
            "\n"
            "  [[12 13]\n"
            "   [14 15]]]]");
}

TEST(NDAPrint, SummarizationRank1) {
  nda::array<long, 1> A(1001);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A), "[   0    1    2 ...  998  999 1000]");

  // exactly at the threshold: no summarization
  nda::array<long, 1> B(1000);
  iota_fill(B);
  EXPECT_EQ(nda::to_string(B).find("..."), std::string::npos);
}

TEST(NDAPrint, SummarizationRank2) {
  nda::array<long, 2> A(40, 40);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A),
            "[[   0    1    2 ...   37   38   39]\n"
            " [  40   41   42 ...   77   78   79]\n"
            " [  80   81   82 ...  117  118  119]\n"
            " ...\n"
            " [1480 1481 1482 ... 1517 1518 1519]\n"
            " [1520 1521 1522 ... 1557 1558 1559]\n"
            " [1560 1561 1562 ... 1597 1598 1599]]");
}

TEST(NDAPrint, SummarizationIsPerAxis) {
  // extent 6 == 2 * edgeitems is kept, extent 200 is truncated
  nda::array<long, 2> A(200, 6);
  iota_fill(A);
  auto const s = nda::to_string(A);
  EXPECT_NE(s.find("\n ...\n"), std::string::npos);
  EXPECT_EQ(s.find("... "), std::string::npos);
}

TEST(NDAPrint, SummarizationOptions) {
  nda::array<long, 1> A(10);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A, {.threshold = 5, .edgeitems = 2}), "[0 1 ... 8 9]");
  EXPECT_EQ(nda::to_string(A, {.threshold = -1}), "[0 1 2 3 4 5 6 7 8 9]");
}

TEST(NDAPrint, DefaultOptionsAreUsedByStreamOperator) {
  nda::array<long, 2> A(40, 40);
  iota_fill(A);
  EXPECT_EQ(via_stream(A), nda::to_string(A));
  EXPECT_NE(via_stream(A).find("..."), std::string::npos);
  // only print_to/to_string can change the options
  EXPECT_EQ(nda::to_string(A, {.threshold = -1}).find("..."), std::string::npos);
}

TEST(NDAPrint, Complex) {
  using cd = std::complex<double>;
  nda::array<cd, 2> A{{cd{0, 1}, cd{2, 3}}, {cd{4, 5}, cd{6, 7}}};
  EXPECT_EQ(nda::to_string(A),
            "[[0+1j 2+3j]\n"
            " [4+5j 6+7j]]");
}

TEST(NDAPrint, ComplexPartsAreAlignedIndependently) {
  using cd = std::complex<double>;
  // numpy: '[  1.+2.j 100.+3.j]'
  EXPECT_EQ(nda::to_string(nda::array<cd, 1>{cd{1, 2}, cd{100, 3}}), "[  1+2j 100+3j]");
  // numpy: '[1.  +2.j 1.+300.j]'
  EXPECT_EQ(nda::to_string(nda::array<cd, 1>{cd{1, 2}, cd{1, 300}}), "[1  +2j 1+300j]");
  // numpy: '[1.-2.j 1.+0.j]'
  EXPECT_EQ(nda::to_string(nda::array<cd, 1>{cd{1, -2}, cd{1, 0}}), "[1-2j 1+0j]");
}

TEST(NDAPrint, ComplexPairStyleOptOut) {
  using cd = std::complex<double>;
  EXPECT_EQ(nda::to_string(nda::array<cd, 1>{cd{0.1, 0.2}}, {.complex_j = false}), "[(0.1,0.2)]");
}

TEST(NDAPrint, EmptyArrays) {
  EXPECT_EQ(nda::to_string(nda::array<long, 1>(0)), "[]");
  EXPECT_EQ(nda::to_string(nda::array<long, 2>(0, 3)), "[]");
  EXPECT_EQ(nda::to_string(nda::array<long, 2>(3, 0)), "[]");
  EXPECT_EQ(nda::to_string(nda::array<long, 3>(2, 0, 3)), "[]");
}

TEST(NDAPrint, StridedViewsAndSlices) {
  nda::array<long, 2> A(3, 4);
  iota_fill(A);
  EXPECT_EQ(nda::to_string(A(nda::range::all, nda::range(0, 4, 2))),
            "[[ 0  2]\n"
            " [ 4  6]\n"
            " [ 8 10]]");
  EXPECT_EQ(nda::to_string(A(1, nda::ellipsis{})), "[4 5 6 7]");
  // to_string agrees with the stream operator, and an array with its view
  EXPECT_EQ(nda::to_string(A), via_stream(A));
  EXPECT_EQ(nda::to_string(A), nda::to_string(nda::array_view<long, 2>(A)));
}

TEST(NDAPrint, IsIndependentOfMemoryLayout) {
  // regression: the old rank > 2 printer dumped the elements in memory order
  nda::array<long, 3> A(2, 2, 3);
  nda::array<long, 3, nda::F_layout> B(2, 2, 3);
  long n = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 3; ++k) {
        A(i, j, k) = n;
        B(i, j, k) = n;
        ++n;
      }
  EXPECT_EQ(nda::to_string(A), nda::to_string(B));
}

TEST(NDAPrint, HangingIndentOption) {
  EXPECT_EQ(nda::to_string(nda::array<long, 2>{{0, 1}, {2, 3}}, {.indent = 4}),
            "[[0 1]\n"
            "     [2 3]]");
}

TEST(NDAPrint, SmallIntegerTypesArePrintedAsNumbers) {
  nda::array<uint8_t, 1> A{1, 200};
  EXPECT_EQ(nda::to_string(A), "[  1 200]");
}

TEST(NDAPrint, NestedArrays) {
  nda::array<nda::array<int, 1>, 1> A(2);
  for (auto &x : A) x = nda::array<int, 1>{1, 2, 3};
  EXPECT_EQ(nda::to_string(A), "[[1 2 3] [1 2 3]]");
}

TEST(NDAPrint, LazyTypesValueViewVsStructuralView) {
  nda::array<long, 1> V{1, 2, 3};

  // nda::to_string prints the values, the stream operator the structure
  EXPECT_EQ(nda::to_string(V + V), "[2 4 6]");
  EXPECT_EQ(via_stream(V + V), "([1 2 3] + [1 2 3])");

  EXPECT_EQ(nda::to_string(-V), "[-1 -2 -3]");
  EXPECT_EQ(via_stream(-V), "-[1 2 3]");

  nda::array<double, 1> D{0.0, 1.0};
  EXPECT_EQ(nda::to_string(nda::map([](double x) { return 2 * x; })(D)), "[0 2]");
  EXPECT_EQ(via_stream(nda::map([](double x) { return 2 * x; })(D)), "mapped");

  auto adapter = nda::array_adapter{std::array<long, 2>{2, 3}, [](long i, long j) { return 3 * i + j; }};
  EXPECT_EQ(nda::to_string(adapter),
            "[[0 1 2]\n"
            " [3 4 5]]");
  EXPECT_EQ(via_stream(adapter), "array_adapter of shape (2 3)");
}

TEST(NDAPrint, MatrixAlgebraExpressionPrintsItsOwnSemantics) {
  // for matrices, scalar + matrix behaves as scalar * identity + matrix
  auto M = nda::matrix<long>{{1, 2}, {3, 4}};
  EXPECT_EQ(nda::to_string(10 + M),
            "[[11  2]\n"
            " [ 3 14]]");
  EXPECT_EQ(nda::to_string(10 + M), nda::to_string(nda::make_regular(10 + M)));
}

TEST(NDAPrint, CustomArrayType) {
  // a type modelling nda::Array without being an nda::MemoryArray is printed element-wise, and in particular is
  // not mistaken for an array in device memory (its address space is mem::None)
  auto a = array_of_rank<long, 2>{{2, 3}};
  EXPECT_EQ(nda::to_string(a),
            "[[2 2 2]\n"
            " [2 2 2]]");
}

#if defined(__cpp_lib_format)

TEST(NDAPrint, FormatterWithoutSpecMatchesStreamOperator) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;
  EXPECT_EQ(std::format("{}", A), via_stream(A));
  EXPECT_EQ(std::format("{:}", A), via_stream(A));

  // guards against rendering the elements with std::format instead of the stream operator: the two differ in the
  // default precision for floating point types
  nda::array<double, 1> D{1.0 / 3, 2.0};
  EXPECT_EQ(std::format("{}", D), via_stream(D));
  EXPECT_NE(std::format("{}", D).find("0.333333"), std::string::npos);
}

TEST(NDAPrint, FormatterPrintsValuesOfLazyTypes) {
  nda::array<long, 1> V{1, 2, 3};

  // in contrast to the stream operator, the formatter prints the values of an expression
  EXPECT_EQ(std::format("{}", V + V), "[2 4 6]");
  EXPECT_EQ(via_stream(V + V), "([1 2 3] + [1 2 3])");

  nda::array<double, 1> D{0.0, 1.0};
  EXPECT_EQ(std::format("{}", nda::map([](double x) { return 2 * x; })(D)), "[0 2]");
  EXPECT_EQ(via_stream(nda::map([](double x) { return 2 * x; })(D)), "mapped");

  auto adapter = nda::array_adapter{std::array<long, 2>{2, 3}, [](long i, long j) { return 3 * i + j; }};
  EXPECT_EQ(std::format("{}", adapter),
            "[[0 1 2]\n"
            " [3 4 5]]");
  EXPECT_EQ(via_stream(adapter), "array_adapter of shape (2 3)");

  auto M = nda::matrix<long>{{1, 2}, {3, 4}};
  EXPECT_EQ(std::format("{}", 10 + M), std::format("{}", nda::make_regular(10 + M)));
}

TEST(NDAPrint, FormatterForwardsSpecToElements) {
  EXPECT_EQ(std::format("{:.3f}", nda::array<double, 1>{1.5, 2.25}), "[1.500 2.250]");
  EXPECT_EQ(std::format("{:03d}", nda::array<long, 2>{{1, 2}, {3, 4}}),
            "[[001 002]\n"
            " [003 004]]");
  EXPECT_EQ(std::format("{:#x}", nda::array<long, 1>{10, 255}), "[ 0xa 0xff]");

  // the spec is applied to the real and the imaginary part separately
  using cd = std::complex<double>;
  EXPECT_EQ(std::format("{:.1f}", nda::array<cd, 1>{cd{0, 1}, cd{2, -3}}), "[0.0+1.0j 2.0-3.0j]");
  // an explicit sign in the spec is not doubled on the imaginary part
  EXPECT_EQ(std::format("{:+.1f}", nda::array<cd, 1>{cd{0, 1}, cd{2, -3}}), "[+0.0+1.0j +2.0-3.0j]");

  // a spec also applies element-wise to lazy types
  nda::array<long, 1> V{1, 2, 3};
  EXPECT_EQ(std::format("{:03d}", V + V), "[002 004 006]");
}

// std::format("{:>{}}", A, 5) is a compile time error: parse() throws during constant evaluation.

#endif // __cpp_lib_format

TEST(NDAPrint, ElementsOfLazyTypesAreEvaluatedExactlyOnce) {
  nda::array<double, 2> A(3, 4);
  iota_fill(A);
  long count = 0;
  auto ex    = nda::map([&count](double x) {
    ++count;
    return x;
  })(A);
  auto s = nda::to_string(ex);
  EXPECT_EQ(count, A.size());
  EXPECT_FALSE(s.empty());
}
