// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides various overloads of the operator<< for nda related objects.
 */

#pragma once

#include "./arithmetic.hpp"
#include "./concepts.hpp"
#include "./array_adapter.hpp"
#include "./layout/idx_map.hpp"
#include "./layout/permutation.hpp"
#include "./map.hpp"
#include "./pretty_print.hpp"
#include "./traits.hpp"

#include <cstdint>
#include <ostream>

#if defined(__cpp_lib_format)
#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#endif

namespace nda {

  /**
   * @addtogroup layout_utils
   * @{
   */

  /**
   * @brief Write an nda::layout_prop_e to a `std::ostream`.
   *
   * @param sout `std::ostream` object.
   * @param p nda::layout_prop_e object.
   * @return Reference to `std::ostream` object.
   */
  inline std::ostream &operator<<(std::ostream &sout, layout_prop_e p) {
    return sout << (has_contiguous(p) ? "contiguous   " : " ") << (has_strided_1d(p) ? "strided_1d   " : " ")
                << (has_smallest_stride_is_one(p) ? "smallest_stride_is_one   " : " ");
  }

  /**
   * @brief Write an nda::idx_map to a `std::ostream`.
   *
   * @tparam Rank Rank of the nda::idx_map.
   * @tparam StaticExtents StaticExtents of the nda::idx_map.
   * @tparam StrideOrder StrideOrder of the nda::idx_map.
   * @tparam LayoutProp Layout property of the nda::idx_map.
   * @param sout `std::ostream` object.
   * @param idxm nda::idx_map object.
   * @return Reference to `std::ostream` object.
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  std::ostream &operator<<(std::ostream &sout, idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &idxm) {
    return sout << "  Lengths  : " << idxm.lengths() << "\n"
                << "  Strides  : " << idxm.strides() << "\n"
                << "  StaticExtents  : " << decode<Rank>(StaticExtents) << "\n"
                << "  MemoryStrideOrder   : " << idxm.stride_order << "\n"
                << "  Flags   :  " << LayoutProp << "\n";
  }

  /** @} */

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Write an nda::basic_array or nda::basic_array_view to a `std::ostream`.
   *
   * @details Uses the rank-generic, numpy-like format of nda::print_to with default nda::print_options. Use
   * nda::print_to directly to print with non-default options.
   *
   * @tparam A Type of the nda::basic_array or nda::basic_array_view.
   * @param sout `std::ostream` object.
   * @param a nda::basic_array or nda::basic_array_view object.
   * @return Reference to `std::ostream` object.
   */
  template <typename A>
  std::ostream &operator<<(std::ostream &sout, A const &a)
    requires(is_regular_or_view_v<A>)
  {
    return print_to(sout, a);
  }

  /**
   * @brief Write an nda::array_adapter to a `std::ostream`.
   *
   * @tparam R Rank of the nda::array_adapter.
   * @tparam F Callable type of the nda::array_adapter.
   * @param sout `std::ostream` object.
   * @param aa nda::array_adapter object.
   * @return Reference to `std::ostream` object.
   */
  template <int R, typename F>
  std::ostream &operator<<(std::ostream &sout, array_adapter<R, F> const &aa) {
    return sout << "array_adapter of shape " << aa.shape();
  }

  /// @cond
  // Forward declarations (necessary for libclang parsing).
  template <char OP, Array A>
  struct expr_unary;

  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;
  /// @endcond

  /**
   * @brief Write an nda::expr_unary to a `std::ostream`.
   *
   * @tparam OP Unary operator.
   * @tparam A nda::Array type.
   * @param sout `std::ostream` object.
   * @param ex nda::expr_unary object.
   * @return Reference to `std::ostream` object.
   */
  template <char OP, Array A>
  std::ostream &operator<<(std::ostream &sout, expr_unary<OP, A> const &ex) {
    return sout << OP << ex.a;
  }

  /**
   * @brief Write an nda::expr to a `std::ostream`.
   *
   * @tparam OP Binary operator.
   * @tparam L nda::ArrayOrScalar type of left hand side.
   * @tparam R nda::ArrayOrScalar type of right hand side.
   * @param sout `std::ostream` object.
   * @param ex nda::expr object.
   * @return Reference to `std::ostream` object.
   */
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  std::ostream &operator<<(std::ostream &sout, expr<OP, L, R> const &ex) {
    return sout << "(" << ex.l << " " << OP << " " << ex.r << ")";
  }

  /**
   * @brief Write an nda::expr_call to a `std::ostream`.
   *
   * @tparam F Callable type.
   * @tparam As Argument types.
   * @param sout `std::ostream` object.
   * @return Reference to `std::ostream` object.
   */
  template <typename F, typename... As>
  std::ostream &operator<<(std::ostream &sout, expr_call<F, As...> const &) {
    return sout << "mapped"; //array<value_type, std::decay_t<A>::rank>(x);
  }

  /** @} */

} // namespace nda

#if defined(__cpp_lib_format)

/**
 * @ingroup av_utils
 * @brief `std::formatter` specialization for nda::Array types.
 *
 * @details Always prints the values of the array element-wise, in the rank-generic, numpy-like format of
 * nda::print_to with default nda::print_options. The format spec is forwarded to the *elements*, i.e.
 * `std::format("{:.3f}", A)` applies `.3f` to every element of `A`. For `std::complex` elements it is applied to the
 * real and the imaginary part separately, since `std::formatter` is not specialized for `std::complex`.
 *
 * Note that this also formats the lazy expression types, in which case the values of the expression are printed
 * rather than its structure. This differs from nda::operator<<, which prints the structure of an expression.
 *
 * Dynamic width and precision arguments, e.g. `"{:>{}}"`, are not supported and lead to a compile time error. Only
 * `char` output is supported, i.e. formatting an array into a `std::wstring` is a compile time error.
 *
 * @tparam A nda::Array type to be formatted.
 */
template <nda::Array A>
struct std::formatter<A, char> {
  private:
  // Format spec applied to the elements. This is a view into the format string, which outlives the format call.
  std::string_view elem_spec_{};

  public:
  /**
   * @brief Parse the format spec, which is applied to the elements of the array.
   *
   * @param ctx Format parse context.
   * @return Iterator past the end of the parsed format spec.
   */
  constexpr auto parse(std::format_parse_context &ctx) {
    auto const first = ctx.begin();
    auto it          = first;
    while (it != ctx.end() and *it != '}') {
      if (*it == '{') throw std::format_error("Error in std::formatter<nda::Array>: Dynamic width or precision arguments are not supported");
      ++it;
    }
    elem_spec_ = std::string_view(first, it);
    if (not elem_spec_.empty()) {
      if constexpr (not nda::detail::has_std_formatter_v<typename nda::remove_complex<nda::get_value_t<A>>::type>)
        throw std::format_error("Error in std::formatter<nda::Array>: Element type is not formattable, only \"{}\" is supported");
    }
    return it;
  }

  /**
   * @brief Format an nda::Array into the output of the format context.
   *
   * @param a nda::Array object to format.
   * @param ctx Format context to write the output to.
   * @return Iterator past the end of the written output.
   */
  template <typename FmtCtx>
  auto format(A const &a, FmtCtx &ctx) const {
    // With an empty spec, the elements are rendered with the stream operator, exactly as nda::operator<< does. This
    // is what makes std::format("{}", a) and (std::ostringstream{} << a) agree for arrays and views.
    auto const s = elem_spec_.empty() ?
       nda::to_string(a) :
       nda::detail::print_to_string(a, nda::print_options{}, nda::detail::format_renderer{"{:" + std::string{elem_spec_} + "}"});
    return std::copy(s.begin(), s.end(), ctx.out());
  }
};

#endif // __cpp_lib_format
