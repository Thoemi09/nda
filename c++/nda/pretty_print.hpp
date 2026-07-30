// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a rank-generic, numpy-like pretty printer for nda::Array types.
 */

#pragma once

#include "./concepts.hpp"
#include "./macros.hpp"
#include "./mem/address_space.hpp"
#include "./stdutil/array.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__cpp_lib_format)
#include <format>
#include <iterator>
#endif

namespace nda {

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Options controlling the numpy-like pretty printing of nda::Array types.
   *
   * @details These are a plain parameter of nda::to_string and nda::print_to. There is deliberately no global
   * default instance: nda::operator<< and the `std::formatter` specialization always use a default constructed
   * nda::print_options.
   */
  struct print_options {
    /// Total number of elements above which the output is summarized with `...`. Negative means never summarize.
    long threshold = 1000;

    /// Number of leading/trailing items kept along each summarized dimension.
    long edgeitems = 3;

    /// Number of additional spaces prepended to every line but the first (hanging indent).
    int indent = 0;

    /// Render `std::complex` values numpy-like as `re+imj`. If false, the value is rendered as a whole, i.e. `(re,im)`.
    bool complex_j = true;
  };

  namespace detail {

    // Which indices of an array are printed along each of its dimensions, and where the "..." placeholders go. The
    // printed indices are not stored: the only per-dimension input is the extent, everything else is derived from the
    // single number of edge items below.
    template <int R>
    struct print_geometry {
      // Shape of the array.
      std::array<long, R> shape;

      // Number of leading/trailing indices kept along every truncated dimension, or -1 if nothing is truncated.
      long edge = -1;

      // numpy semantics: summarize if and only if the total number of elements exceeds the threshold, and then
      // truncate every dimension whose extent exceeds 2 * edgeitems.
      constexpr print_geometry(std::array<long, R> const &sh, long total, print_options const &opts) : shape(sh) {
        if ((opts.threshold >= 0) and (total > opts.threshold) and (opts.edgeitems > 0)) edge = opts.edgeitems;
      }

      // Extent of dimension k. The single place where a dimension is used to index into a std::array.
      [[nodiscard]] constexpr long extent(int k) const { return shape[static_cast<size_t>(k)]; }

      // True if dimension k is truncated.
      [[nodiscard]] constexpr bool cut(int k) const { return edge > 0 and extent(k) > 2 * edge; }

      // Number of indices printed along dimension k.
      [[nodiscard]] constexpr long size(int k) const { return cut(k) ? 2 * edge : extent(k); }

      // The p-th printed index along dimension k, with 0 <= p < size(k).
      [[nodiscard]] constexpr long idx(int k, long p) const { return (cut(k) and p >= edge) ? extent(k) - 2 * edge + p : p; }

      // Position among the printed indices of dimension k before which the "..." is printed, or -1 if the dimension
      // is not truncated. Always >= 1 when it is, hence always preceded by a separator.
      [[nodiscard]] constexpr long ellipsis_pos(int k) const { return cut(k) ? edge : -1; }
    };

    // A rendered element: the whole value in re, or its real part with the signed imaginary part including the
    // trailing 'j' in im, e.g. {"1", "+2j"}. The two parts are aligned independently, numpy-like.
    struct token {
      std::string re;
      std::string im;
    };

    // Turn a rendered imaginary part into its token: prepend a '+' unless it already carries a sign and append the
    // 'j'. The sign is part of the token so that a user supplied format spec composes with it, i.e. "{:+.3f}" does
    // not lead to a doubled sign.
    inline std::string imag_token(std::string s) {
      auto pos = s.find_first_not_of(' ');
      if (pos == std::string::npos) pos = 0;
      if (s.empty() or (s[pos] != '+' and s[pos] != '-')) s.insert(pos, 1, '+');
      s.push_back('j');
      return s;
    }

    // Visit every printed element of the given array in row-major order. This is the single definition of the element
    // order shared by both passes of the printer. It only uses shape() and operator(), so it works for every
    // nda::Array including the lazy expression types.
    template <int R, Array A, typename F>
    void for_each_printed(A const &a, print_geometry<R> const &g, F const &f) {
      std::array<long, R> idx{};
      auto rec = [&](auto &self, int k) -> void {
        for (long p = 0, n = g.size(k); p < n; ++p) {
          idx[static_cast<size_t>(k)] = g.idx(k, p);
          if (k + 1 == R)
            std::apply([&a, &f](auto... is) { f(a(is...)); }, idx);
          else
            self(self, k + 1);
        }
      };
      rec(rec, 0);
    }

    // Rendered elements of an array, together with the field widths they have to be aligned to.
    struct print_tokens {
      std::vector<token> tok;

      size_t w_re = 0;
      size_t w_im = 0;

      // True if some token spans several lines, e.g. for an array of arrays, in which case padding is meaningless
      // and therefore disabled.
      bool multiline = false;
    };

    // First pass: render every printed element and determine the widths to align them to.
    template <int R, Array A, typename Render>
    print_tokens render_tokens(A const &a, print_geometry<R> const &g, Render const &render) {
      print_tokens tk;
      size_t n = 1;
      for (int k = 0; k < R; ++k) n *= static_cast<size_t>(g.size(k));
      tk.tok.reserve(n);

      for_each_printed<R>(a, g, [&tk, &render](auto const &x) { tk.tok.push_back(render(x)); });

      for (auto const &t : tk.tok) {
        tk.w_re      = std::max(tk.w_re, t.re.size());
        tk.w_im      = std::max(tk.w_im, t.im.size());
        tk.multiline = tk.multiline or (t.re.find('\n') != std::string::npos) or (t.im.find('\n') != std::string::npos);
      }
      return tk;
    }

    // Second pass: emit the brackets, separators and padded tokens. The separator between two sub-blocks along
    // dimension k of a rank R array consists of (R - 1 - k) newlines followed by (indent + k + 1) spaces of hanging
    // indent, while the elements of the innermost dimension are separated by a single space.
    template <int R>
    void emit(std::string &out, print_tokens const &tk, print_geometry<R> const &g, print_options const &opts, int k, size_t &cursor) {
      bool const leaf = (k == R - 1);
      auto separator  = [&out, &opts, leaf, k] {
        if (leaf) {
          out.push_back(' ');
        } else {
          long const n_newlines = R - 1 - k;
          long const n_spaces   = opts.indent + k + 1;
          out.append(static_cast<size_t>(n_newlines), '\n');
          out.append(static_cast<size_t>(n_spaces), ' ');
        }
      };

      out.push_back('[');
      for (long p = 0, n = g.size(k); p < n; ++p) {
        if (p != 0) separator();
        if (p == g.ellipsis_pos(k)) {
          out.append("...");
          separator();
        }
        if (not leaf) {
          emit<R>(out, tk, g, opts, k + 1, cursor);
          continue;
        }
        auto const &t = tk.tok[cursor++];
        if (not tk.multiline) out.append(tk.w_re - t.re.size(), ' ');
        out.append(t.re);
        if (not t.im.empty()) {
          if (not tk.multiline) out.append(tk.w_im - t.im.size(), ' ');
          out.append(t.im);
        }
      }
      out.push_back(']');
    }

    // Render an array into a string using the given element renderer.
    template <Array A, typename Render>
    std::string print_to_string(A const &a, print_options const &opts, Render const &render) {
      constexpr int R = get_rank<A>;
      static_assert(R > 0, "Error in nda::detail::print_to_string: Printing is only defined for arrays of rank > 0");

      // Elements in device memory cannot be read on the host. Both halves of the condition matter: a lazy expression
      // over device arrays is not an nda::MemoryArray, and the default address space of an unknown type is None.
      if constexpr (mem::have_device_compatible_addr_space<A> and not mem::have_host_compatible_addr_space<A>) {
        return "<nda array of shape " + std::to_string(a.shape()) + " in device memory>";
      } else {
        if (a.size() == 0) return "[]";
        auto const g  = print_geometry<R>(a.shape(), a.size(), opts);
        auto const tk = render_tokens<R>(a, g, render);
        std::string out;
        size_t cursor = 0;
        emit<R>(out, tk, g, opts, 0, cursor);
        ENSURES(cursor == tk.tok.size());
        return out;
      }
    }

    // Element renderer using the stream operator. Available without <format> support.
    struct ostream_renderer {
      bool complex_j = true;

      template <typename U>
      token operator()(U const &x) const {
        if constexpr (is_complex_v<U>) {
          if (complex_j) return {str(x.real()), imag_token(str(x.imag()))};
        }
        return {str(x), {}};
      }

      private:
      template <typename U>
      static std::string str(U const &x) {
        std::ostringstream os;
        // Render (un)signed chars as numbers: streaming them as characters would break the alignment.
        if constexpr (std::is_same_v<U, signed char> or std::is_same_v<U, unsigned char>)
          os << static_cast<int>(x);
        else
          os << x;
        return os.str();
      }
    };

#if defined(__cpp_lib_format)

    // True if std::formatter<U, char> is enabled. The primary template of std::formatter deletes its constructors,
    // which makes this a C++20 compatible stand-in for the C++23 std::formattable concept.
    template <typename U>
    inline constexpr bool has_std_formatter_v = std::is_default_constructible_v<std::formatter<std::remove_cvref_t<U>, char>>;

    // Element renderer applying a runtime format spec to each element.
    struct format_renderer {
      // Full format string, e.g. "{:.3f}".
      std::string fmt;

      template <typename U>
      token operator()(U const &x) const {
        // std::formatter is not specialized for std::complex in C++20/23, so the spec is always applied to the real
        // and the imaginary part separately.
        if constexpr (is_complex_v<U>)
          return {str(x.real()), imag_token(str(x.imag()))};
        else
          return {str(x), {}};
      }

      private:
      template <typename U>
      std::string str(U const &x) const {
        // std::make_format_args takes 'const Args&...' in C++20 as published, but 'Args&...' since P2418R2 and in
        // C++23. A non-const lvalue is what both signatures accept.
        auto v = x;
        std::string res;
        std::vformat_to(std::back_inserter(res), fmt, std::make_format_args(v));
        return res;
      }
    };

#endif // __cpp_lib_format

  } // namespace detail

  /**
   * @brief Get a numpy-like string representation of an nda::Array.
   *
   * @details In contrast to nda::operator<<, this also works for the lazy expression types, in which case the
   * values of the expression are printed instead of its structure. Every printed element is evaluated exactly once.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @param opts nda::print_options.
   * @return `std::string` representation of the array.
   */
  template <Array A>
  std::string to_string(A const &a, print_options const &opts = {}) {
    return detail::print_to_string(a, opts, detail::ostream_renderer{opts.complex_j});
  }

  /**
   * @brief Write a numpy-like string representation of an nda::Array to a `std::ostream`.
   *
   * @details In contrast to nda::operator<<, this also works for the lazy expression types, in which case the
   * values of the expression are printed instead of its structure. Every printed element is evaluated exactly once.
   *
   * @tparam A nda::Array type.
   * @param out `std::ostream` object.
   * @param a nda::Array object.
   * @param opts nda::print_options.
   * @return Reference to `std::ostream` object.
   */
  template <Array A>
  std::ostream &print_to(std::ostream &out, A const &a, print_options const &opts = {}) {
    return out << to_string(a, opts);
  }

  /** @} */

} // namespace nda
