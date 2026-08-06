// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/stdutil/format.hpp>

#include <complex>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

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

// --------------------------------------------------------------------------------------------------------------
// The stream operator, which is deliberately left in its old, comma separated form. Only std::format gives the
// numpy-like output below.
// --------------------------------------------------------------------------------------------------------------

TEST(NDAPrint, StreamOperatorRank1) { EXPECT_EQ(via_stream(nda::array<long, 1>{1, 2, 3}), "[1,2,3]"); }

TEST(NDAPrint, StreamOperatorRank2) {
  EXPECT_EQ(via_stream(nda::array<long, 2>{{0, 1, 2}, {3, 4, 5}}),
            "\n[[0,1,2]\n"
            " [3,4,5]]");
}

TEST(NDAPrint, StreamOperatorRank3) {
  nda::array<long, 3> A(2, 2, 2);
  iota_fill(A);
  EXPECT_EQ(via_stream(A), "[0,1,2,3,4,5,6,7]");
}

TEST(NDAPrint, StreamOperatorPrintsTheStructureOfLazyTypes) {
  nda::array<long, 1> V{1, 2, 3};
  EXPECT_EQ(via_stream(V + V), "([1,2,3] + [1,2,3])");
  EXPECT_EQ(via_stream(-V), "-[1,2,3]");
  EXPECT_EQ(via_stream(nda::map([](long x) { return 2 * x; })(V)), "mapped");
  EXPECT_EQ(via_stream(nda::array_adapter{std::array<long, 2>{2, 3}, [](long i, long j) { return 3 * i + j; }}), "array_adapter of shape (2 3)");
}

#if defined(__cpp_lib_format)

// --------------------------------------------------------------------------------------------------------------
// std::format. All expected strings below have been validated against numpy 1.26.4, except where nda deliberately
// diverges (no uniform float formatting, complex values as "(re+imi)", and no line wrapping).
// --------------------------------------------------------------------------------------------------------------

TEST(NDAFormat, Rank1) { EXPECT_EQ(std::format("{}", nda::array<long, 1>{1, 2, 3}), "[1 2 3]"); }

TEST(NDAFormat, Rank1RightAlignedToCommonWidth) { EXPECT_EQ(std::format("{}", nda::array<long, 1>{-1, 10, 3}), "[-1 10  3]"); }

TEST(NDAFormat, Rank2) {
  EXPECT_EQ(std::format("{}", nda::array<long, 2>{{0, 1, 2}, {3, 4, 5}}),
            "[[0 1 2]\n"
            " [3 4 5]]");
}

TEST(NDAFormat, Rank2CommonWidth) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;
  EXPECT_EQ(std::format("{}", A),
            "[[ 0  1  2]\n"
            " [10 11 12]]");
}

TEST(NDAFormat, Rank3) {
  nda::array<long, 3> A(2, 2, 3);
  iota_fill(A);
  EXPECT_EQ(std::format("{}", A),
            "[[[ 0  1  2]\n"
            "  [ 3  4  5]]\n"
            "\n"
            " [[ 6  7  8]\n"
            "  [ 9 10 11]]]");
}

TEST(NDAFormat, Rank4) {
  nda::array<long, 4> A(2, 2, 2, 2);
  iota_fill(A);
  EXPECT_EQ(std::format("{}", A),
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

TEST(NDAFormat, SummarizationRank1) {
  nda::array<long, 1> A(1001);
  iota_fill(A);
  EXPECT_EQ(std::format("{}", A), "[   0    1    2 ...  998  999 1000]");

  // exactly at the threshold: no summarization
  nda::array<long, 1> B(1000);
  iota_fill(B);
  EXPECT_EQ(std::format("{}", B).find("..."), std::string::npos);
}

TEST(NDAFormat, SummarizationRank2) {
  nda::array<long, 2> A(40, 40);
  iota_fill(A);
  EXPECT_EQ(std::format("{}", A),
            "[[   0    1    2 ...   37   38   39]\n"
            " [  40   41   42 ...   77   78   79]\n"
            " [  80   81   82 ...  117  118  119]\n"
            " ...\n"
            " [1480 1481 1482 ... 1517 1518 1519]\n"
            " [1520 1521 1522 ... 1557 1558 1559]\n"
            " [1560 1561 1562 ... 1597 1598 1599]]");
}

TEST(NDAFormat, SummarizationIsPerAxis) {
  // extent 6 == 2 * edgeitems is kept, extent 200 is truncated
  nda::array<long, 2> A(200, 6);
  iota_fill(A);
  auto const s = std::format("{}", A);
  EXPECT_NE(s.find("\n ...\n"), std::string::npos);
  EXPECT_EQ(s.find("... "), std::string::npos);
}

TEST(NDAFormat, EmptyArrays) {
  EXPECT_EQ(std::format("{}", nda::array<long, 1>(0)), "[]");
  EXPECT_EQ(std::format("{}", nda::array<long, 2>(0, 3)), "[]");
  EXPECT_EQ(std::format("{}", nda::array<long, 2>(3, 0)), "[]");
  EXPECT_EQ(std::format("{}", nda::array<long, 3>(2, 0, 3)), "[]");
}

TEST(NDAFormat, StridedViewsAndSlices) {
  nda::array<long, 2> A(3, 4);
  iota_fill(A);
  EXPECT_EQ(std::format("{}", A(nda::range::all, nda::range(0, 4, 2))),
            "[[ 0  2]\n"
            " [ 4  6]\n"
            " [ 8 10]]");
  EXPECT_EQ(std::format("{}", A(1, nda::ellipsis{})), "[4 5 6 7]");
  EXPECT_EQ(std::format("{}", A), std::format("{}", nda::array_view<long, 2>(A)));
}

TEST(NDAFormat, IsIndependentOfMemoryLayout) {
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
  EXPECT_EQ(std::format("{}", A), std::format("{}", B));
}

TEST(NDAFormat, SmallIntegerTypesAreFormattedAsNumbers) { EXPECT_EQ(std::format("{}", nda::array<uint8_t, 1>{1, 200}), "[  1 200]"); }

TEST(NDAFormat, NestedArrays) {
  nda::array<nda::array<int, 1>, 1> A(2);
  for (auto &x : A) x = nda::array<int, 1>{1, 2, 3};
  EXPECT_EQ(std::format("{}", A), "[[1 2 3] [1 2 3]]");
}

TEST(NDAFormat, CustomArrayType) {
  // a type modelling nda::Array without being an nda::MemoryArray is formatted element-wise, and in particular is
  // not mistaken for an array in device memory (its address space is mem::None)
  auto a = array_of_rank<long, 2>{{2, 3}};
  EXPECT_EQ(std::format("{}", a),
            "[[2 2 2]\n"
            " [2 2 2]]");
}

TEST(NDAFormat, PrintsTheValuesOfLazyTypes) {
  nda::array<long, 1> V{1, 2, 3};
  EXPECT_EQ(std::format("{}", V + V), "[2 4 6]");
  EXPECT_EQ(std::format("{}", -V), "[-1 -2 -3]");
  EXPECT_EQ(std::format("{}", nda::map([](long x) { return 2 * x; })(V)), "[2 4 6]");
  EXPECT_EQ(std::format("{}", nda::array_adapter{std::array<long, 2>{2, 3}, [](long i, long j) { return 3 * i + j; }}),
            "[[0 1 2]\n"
            " [3 4 5]]");

  // for matrices, scalar + matrix behaves as scalar * identity + matrix
  auto M = nda::matrix<long>{{1, 2}, {3, 4}};
  EXPECT_EQ(std::format("{}", 10 + M),
            "[[11  2]\n"
            " [ 3 14]]");
  EXPECT_EQ(std::format("{}", 10 + M), std::format("{}", nda::make_regular(10 + M)));
}

TEST(NDAFormat, ElementsOfLazyTypesAreEvaluatedExactlyOnce) {
  nda::array<double, 2> A(3, 4);
  iota_fill(A);
  long count = 0;
  auto ex    = nda::map([&count](double x) {
    ++count;
    return x;
  })(A);
  auto s     = std::format("{}", ex);
  EXPECT_EQ(count, A.size());
  EXPECT_FALSE(s.empty());
}

// --------------------------------------------------------------------------------------------------------------
// std::complex elements, rendered like fmtlib does except that the parentheses are always there.
// --------------------------------------------------------------------------------------------------------------

using cd = std::complex<double>;

TEST(NDAFormat, Complex) {
  nda::array<cd, 2> A{{cd{0, 1}, cd{2, 3}}, {cd{4, 5}, cd{6, 7}}};
  EXPECT_EQ(std::format("{}", A),
            "[[(0+1i) (2+3i)]\n"
            " [(4+5i) (6+7i)]]");
}

TEST(NDAFormat, ComplexSigns) {
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, -2}, cd{1, 0}}), "[(1-2i) (1+0i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{0, -2}}), "[(0-2i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{-0.0, -0.0}}), "[(-0-0i)]");
}

TEST(NDAFormat, ComplexIsOneAlignedColumn) {
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, 2}, cd{100, 3}}), "[  (1+2i) (100+3i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, 2}, cd{1, 300}}), "[  (1+2i) (1+300i)]");
}

TEST(NDAFormat, ComplexSpecAppliesToBothParts) {
  nda::array<cd, 1> A{cd{0, 1}, cd{2, -3}};
  EXPECT_EQ(std::format("{:.1f}", A), "[(0.0+1.0i) (2.0-3.0i)]");
  // an explicit sign is not doubled on the imaginary part
  EXPECT_EQ(std::format("{:+.1f}", A), "[(+0.0+1.0i) (+2.0-3.0i)]");
  EXPECT_EQ(std::format("{: }", A), "[( 0+1i) ( 2-3i)]");
  EXPECT_EQ(std::format("{:#x}", nda::array<std::complex<int>, 1>{{10, 255}}), "[(0xa+0xffi)]");
}

TEST(NDAFormat, ComplexNonFiniteParts) {
  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, inf}}), "[(1+inf i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, -inf}}), "[(1-inf i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{1, nan}}), "[(1+nan i)]");
  EXPECT_EQ(std::format("{}", nda::array<cd, 1>{cd{inf, 2}}), "[(inf+2i)]");
}

// --------------------------------------------------------------------------------------------------------------
// The format spec of an element.
// --------------------------------------------------------------------------------------------------------------

TEST(NDAFormat, SpecIsForwardedToTheElements) {
  EXPECT_EQ(std::format("{:.3f}", nda::array<double, 1>{1.5, 2.25}), "[1.500 2.250]");
  EXPECT_EQ(std::format("{:03d}", nda::array<long, 2>{{1, 2}, {3, 4}}),
            "[[001 002]\n"
            " [003 004]]");
  EXPECT_EQ(std::format("{:#x}", nda::array<long, 1>{10, 255}), "[ 0xa 0xff]");
  // a spec also applies element-wise to lazy types
  EXPECT_EQ(std::format("{:03d}", nda::array<long, 1>{1, 2} + nda::array<long, 1>{1, 2}), "[002 004]");
}

TEST(NDAFormat, WidthAndAlignApplyToTheWholeElement) {
  EXPECT_EQ(std::format("{:>10}", nda::array<double, 1>{1.5, 22.25}), "[       1.5      22.25]");
  EXPECT_EQ(std::format("{:<6}", nda::array<long, 1>{1, 22}), "[1      22    ]");
  EXPECT_EQ(std::format("{:*^7}", nda::array<long, 1>{1, 22}), "[***1*** **22***]");

  // the whole "(1+2i)" is padded, not its parts
  EXPECT_EQ(std::format("{:>12}", nda::array<cd, 1>{cd{1, 2}, cd{3, -4}}), "[      (1+2i)       (3-4i)]");

  // a width below the natural column width changes nothing
  EXPECT_EQ(std::format("{:>2}", nda::array<long, 1>{1, 22}), std::format("{}", nda::array<long, 1>{1, 22}));
}

TEST(NDAFormat, ZeroFlagIsSignAware) {
  EXPECT_EQ(std::format("{:08.2f}", nda::array<double, 1>{-1.5, 22.25}), "[-0001.50 00022.25]");

  // for a complex element the zeros pad the whole "(re+imi)", as they do in fmtlib
  EXPECT_EQ(std::format("{:014.2f}", nda::array<cd, 1>{cd{1, 2}, cd{3, -4}}), "[00(1.00+2.00i) 00(3.00-4.00i)]");

  // an explicit alignment makes the flag ignored
  EXPECT_EQ(std::format("{:>08.2f}", nda::array<double, 1>{-1.5}), "[   -1.50]");
}

TEST(NDAFormat, DynamicWidthAndPrecision) {
  nda::array<double, 1> A{1.5, 22.25};
  EXPECT_EQ(std::format("{:>{}}", A, 10), std::format("{:>10}", A));
  EXPECT_EQ(std::format("{:.{}f}", A, 3), std::format("{:.3f}", A));
  EXPECT_EQ(std::format("{:>{}.{}f}", A, 10, 1), std::format("{:>10.1f}", A));
  // manual argument ids
  EXPECT_EQ(std::format("{1:.{0}f}", 3, A), std::format("{:.3f}", A));
}

TEST(NDAFormat, DynamicArgumentsMustBeNonNegativeIntegers) {
  nda::array<double, 1> A{1.5};
  std::string s = "x";
  int neg       = -3;
  EXPECT_THROW({ [[maybe_unused]] auto out = std::vformat("{:>{}}", std::make_format_args(A, s)); }, std::format_error);
  EXPECT_THROW({ [[maybe_unused]] auto out = std::vformat("{:>{}}", std::make_format_args(A, neg)); }, std::format_error);
}

// std::format("{:q}", A), std::format("{:.2d}", A) and std::format("{:#+}", A) are compile time errors: the spec of
// the elements is validated by the formatter of their component type, and so cannot be asserted here.

// --------------------------------------------------------------------------------------------------------------
// nda::print_options, passed by value with nda::with_opts or nda::format_with_opts.
// --------------------------------------------------------------------------------------------------------------

TEST(NDAFormat, SummarizationOptions) {
  nda::array<long, 1> A(10);
  iota_fill(A);
  EXPECT_EQ(nda::format_with_opts({.threshold = 5, .edgeitems = 2}, "{}", A), "[0 1 ... 8 9]");
  EXPECT_EQ(nda::format_with_opts({.threshold = -1}, "{}", A), "[0 1 2 3 4 5 6 7 8 9]");

  nda::array<long, 2> B(40, 40);
  iota_fill(B);
  EXPECT_NE(std::format("{}", B).find("..."), std::string::npos);
  EXPECT_EQ(nda::format_with_opts({.threshold = -1}, "{}", B).find("..."), std::string::npos);
}

TEST(NDAFormat, HangingIndentOption) {
  EXPECT_EQ(nda::format_with_opts({.indent = 4}, "{}", nda::array<long, 2>{{0, 1}, {2, 3}}),
            "[[0 1]\n"
            "     [2 3]]");
}

TEST(NDAFormat, OptionsCanBeAttachedToASingleArgument) {
  nda::array<long, 1> A(10);
  iota_fill(A);
  auto const opts = nda::print_options{.threshold = 5, .edgeitems = 2};
  EXPECT_EQ(std::format("{}", nda::with_opts(A, opts)), nda::format_with_opts(opts, "{}", A));
  EXPECT_EQ(std::format("{:03d}", nda::with_opts(A, opts)), "[000 001 ... 008 009]");

  std::string out;
  std::format_to(std::back_inserter(out), "{}", nda::with_opts(A, opts));
  EXPECT_EQ(out, "[0 1 ... 8 9]");
}

TEST(NDAFormat, FormatWithOptsPassesOtherArgumentsThrough) {
  nda::array<long, 1> A{1, 2, 3};
  EXPECT_EQ(nda::format_with_opts({}, "{}: {} and {:.2f}", "A", A, 1.5), "A: [1 2 3] and 1.50");
  // and applies the options to every array argument
  EXPECT_EQ(nda::format_with_opts({.threshold = 2, .edgeitems = 1}, "{} {}", A, A), "[1 ... 3] [1 ... 3]");
}

#endif // __cpp_lib_format
