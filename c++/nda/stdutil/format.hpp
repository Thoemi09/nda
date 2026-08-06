// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides `std::format` support for nda::Array types, i.e. a rank-generic, numpy-like formatter.
 */

#ifndef STDUTILS_FORMAT_H
#define STDUTILS_FORMAT_H

#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"
#include "./array.hpp"

#if defined(__cpp_lib_format)

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Options controlling the numpy-like formatting of nda::Array types.
   *
   * @details These are not part of the format spec but are passed as a value, either per argument with
   * nda::with_opts or for a whole format call with nda::format_with_opts. There is deliberately no global default
   * instance: the `std::formatter` specialization for nda::Array types always uses default constructed options.
   */
  struct print_options {
    /// Total number of elements above which the output is summarized with `...`. Negative means never summarize.
    long threshold = 1000;

    /// Number of leading/trailing items kept along each summarized dimension.
    long edgeitems = 3;

    /// Number of additional spaces prepended to every line but the first (hanging indent).
    int indent = 0;
  };

  /// @cond
  // Forward declaration.
  template <Array A>
  struct array_with_opts;
  /// @endcond

  namespace detail {

    // True if std::formatter<U, char> is enabled. The primary template of std::formatter deletes its
    // constructors, which makes this a C++20 compatible stand-in for the C++23 std::formattable concept.
    template <typename U>
    inline constexpr bool has_std_formatter_v = std::is_default_constructible_v<std::formatter<std::remove_cvref_t<U>, char>>;

    // Number of leading/trailing indices printed along every truncated dimension, or -1 if nothing is truncated.
    // numpy semantics: summarize if and only if the total number of elements exceeds the threshold, and then
    // truncate every dimension whose extent exceeds 2 * edgeitems.
    constexpr long print_edge(long total, print_options const &opts) {
      return (opts.threshold >= 0 and total > opts.threshold and opts.edgeitems > 0) ? opts.edgeitems : -1;
    }

    // Visit the indices printed along a dimension of extent n, in order. The bool tells the caller that the "..."
    // placeholder goes before the given index. This is the single place that knows the truncation rule.
    template <typename F>
    void for_each_printed_idx(long n, long edge, F const &f) {
      if (edge <= 0 or n <= 2 * edge) {
        for (long i = 0; i < n; ++i) f(i, false);
      } else {
        for (long i = 0; i < edge; ++i) f(i, false);
        for (long i = n - edge; i < n; ++i) f(i, i == n - edge);
      }
    }

    // Visit every printed element of the given array in row-major order. This is the single definition of the
    // element order shared by both passes of the printer. It only uses shape() and operator(), so it works for
    // every nda::Array including the lazy expression types.
    template <int R, Array A, typename F>
    void for_each_printed(A const &a, long edge, F const &f) {
      auto const shape = a.shape();
      std::array<long, R> idx{};
      auto rec = [&](auto &self, int k) -> void {
        for_each_printed_idx(shape[static_cast<size_t>(k)], edge, [&](long i, bool) {
          idx[static_cast<size_t>(k)] = i;
          if (k + 1 == R)
            std::apply([&a, &f](auto... is) { f(a(is...)); }, idx);
          else
            self(self, k + 1);
        });
      };
      rec(rec, 0);
    }

    // The format spec of an element, split into the part that is applied to the whole element and the part that is
    // applied to its components. Everything after the width is carried through verbatim and never interpreted.
    struct elem_spec {
      // Fill character and alignment of the whole element, with align == 0 meaning unspecified, i.e. right.
      char fill  = ' ';
      char align = 0;

      // The '0' flag, i.e. fill with '0' and align right.
      bool zero = false;

      // Static width of the whole element, or 0 if there is none.
      long width = 0;

      // Argument ids of a dynamic width and precision, or -1 if they are static.
      int width_arg = -1;
      int prec_arg  = -1;

      // Sign and '#', and static precision, 'L' and type. Both are applied to the components of an element and
      // are views into the format string, which outlives the format call.
      std::string_view head{};
      std::string_view tail{};
    };

    // Parse a non-negative integer at the beginning of s and remove it from s.
    constexpr long parse_int(std::string_view &s) {
      long v = 0;
      while (not s.empty() and s.front() >= '0' and s.front() <= '9') {
        v = 10 * v + (s.front() - '0');
        s.remove_prefix(1);
      }
      return v;
    }

    // Parse a dynamic argument id, i.e. "{}" or "{n}", at the beginning of s, remove it from s and register it
    // with the parse context. Expects s to start with '{'.
    constexpr int parse_arg_id(std::string_view &s, std::format_parse_context &ctx) {
      s.remove_prefix(1);
      int id = 0;
      if (not s.empty() and s.front() == '}') {
        id = static_cast<int>(ctx.next_arg_id());
      } else {
        id = static_cast<int>(parse_int(s));
        ctx.check_arg_id(static_cast<size_t>(id));
      }
      if (s.empty() or s.front() != '}')
        throw std::format_error("Error in std::formatter<nda::Array>: Unterminated dynamic width or precision argument");
      s.remove_prefix(1);
      return id;
    }

    // Split the format spec of an element: the fill and align, the '0' flag and the width are applied to the whole
    // element, everything else is handed to the formatter of the component type.
    constexpr elem_spec split_elem_spec(std::string_view s, std::format_parse_context &ctx) {
      auto es       = elem_spec{};
      auto is_align = [](char c) { return c == '<' or c == '>' or c == '^'; };
      auto is_flag  = [](char c) { return c == '+' or c == '-' or c == ' ' or c == '#'; };

      // fill-and-align
      if (s.size() >= 2 and is_align(s[1])) {
        es.fill  = s[0];
        es.align = s[1];
        s.remove_prefix(2);
      } else if (not s.empty() and is_align(s[0])) {
        es.align = s[0];
        s.remove_prefix(1);
      }

      // sign and '#' are applied to the components, but precede the width in the spec
      auto const head = s;
      while (not s.empty() and is_flag(s.front())) s.remove_prefix(1);
      es.head = head.substr(0, head.size() - s.size());

      // the '0' flag
      if (not s.empty() and s.front() == '0') {
        es.zero = true;
        s.remove_prefix(1);
      }

      // width
      if (not s.empty() and s.front() == '{')
        es.width_arg = parse_arg_id(s, ctx);
      else
        es.width = parse_int(s);

      // only a dynamic precision has to be recognized here, its value is substituted when formatting
      if (s.size() >= 2 and s[0] == '.' and s[1] == '{') {
        s.remove_prefix(1);
        es.prec_arg = parse_arg_id(s, ctx);
      }

      // the rest, i.e. a static precision, 'L' and the type, is carried through verbatim
      es.tail = s;
      return es;
    }

    // Turn the component spec into the one for an imaginary part: the sign is forced to '+' so that the parts
    // always join, i.e. "(1+2i)" and "(1-2i)", and a user supplied sign is not doubled.
    inline std::string imag_head(std::string_view head) {
      auto res = std::string{head};
      if (not res.empty() and (res[0] == '+' or res[0] == '-' or res[0] == ' '))
        res[0] = '+';
      else
        res.insert(res.begin(), '+');
      return res;
    }

    // Render a single value with the given format string.
    template <typename U>
    std::string format_value(U const &x, std::string const &fmt) {
      // Render (un)signed chars as numbers: formatting them as characters would break the alignment.
      if constexpr (std::is_same_v<U, signed char> or std::is_same_v<U, unsigned char>) {
        return format_value(static_cast<int>(x), fmt);
      } else {
        // std::make_format_args takes 'const Args&...' in C++20 as published, but 'Args&...' since P2418R2 and in
        // C++23. A non-const lvalue is what both signatures accept.
        auto v = x;
        std::string res;
        std::vformat_to(std::back_inserter(res), fmt, std::make_format_args(v));
        return res;
      }
    }

    // Render one element. std::complex values are rendered like fmtlib does, as "(re+imi)", except that the
    // parentheses are always there. A space precedes the 'i' if the imaginary part is not finite, e.g. "(1+inf i)".
    template <typename U>
    std::string render_element(U const &x, std::string const &re_fmt, std::string const &im_fmt) {
      if constexpr (is_complex_v<U>) {
        auto res = "(" + format_value(x.real(), re_fmt) + format_value(x.imag(), im_fmt);
        if (not std::isfinite(x.imag())) res += ' ';
        return res + "i)";
      } else {
        return format_value(x, re_fmt);
      }
    }

    // Rendered elements of an array, together with the field width they are aligned to.
    struct print_tokens {
      std::vector<std::string> tok;

      size_t width = 0;

      // True if some token spans several lines, e.g. for an array of arrays, in which case padding is meaningless
      // and therefore disabled.
      bool multiline = false;
    };

    // First pass: render every printed element and determine the width they are aligned to.
    template <int R, Array A, typename Render>
    print_tokens render_tokens(A const &a, long edge, Render const &render) {
      auto tk = print_tokens{};
      for_each_printed<R>(a, edge, [&tk, &render](auto const &x) { tk.tok.push_back(render(x)); });
      for (auto const &t : tk.tok) {
        tk.width     = std::max(tk.width, t.size());
        tk.multiline = tk.multiline or (t.find('\n') != std::string::npos);
      }
      return tk;
    }

    // Second pass: emit the brackets, separators and padded tokens. The separator between two sub-blocks along
    // dimension k of a rank R array consists of (R - 1 - k) newlines followed by (indent + k + 1) spaces of hanging
    // indent, while the elements of the innermost dimension are separated by a single space.
    template <int R>
    void emit(std::string &out, print_tokens const &tk, std::array<long, R> const &shape, long edge, print_options const &opts, int k,
              size_t &cursor) {
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
      bool first = true;
      for_each_printed_idx(shape[static_cast<size_t>(k)], edge, [&](long, bool ellipsis) {
        if (not first) separator();
        first = false;
        if (ellipsis) {
          out.append("...");
          separator();
        }
        if (not leaf) {
          emit<R>(out, tk, shape, edge, opts, k + 1, cursor);
          return;
        }
        auto const &t = tk.tok[cursor++];
        if (not tk.multiline) out.append(tk.width - t.size(), ' ');
        out.append(t);
      });
      out.push_back(']');
    }

    // Render an array into a string using the given element renderer.
    template <Array A, typename Render>
    std::string render_array(A const &a, print_options const &opts, Render const &render) {
      constexpr int R = get_rank<A>;
      static_assert(R > 0, "Error in nda::detail::render_array: Formatting is only defined for arrays of rank > 0");

      // Elements in device memory cannot be read on the host. Both halves of the condition matter: a lazy expression
      // over device arrays is not an nda::MemoryArray, and the default address space of an unknown type is None.
      if constexpr (mem::have_device_compatible_addr_space<A> and not mem::have_host_compatible_addr_space<A>) {
        return "<nda array of shape " + std::to_string(a.shape()) + " in device memory>";
      } else {
        if (a.size() == 0) return "[]";
        auto const edge = print_edge(a.size(), opts);
        auto const tk   = render_tokens<R>(a, edge, render);
        auto out        = std::string{};
        size_t cursor   = 0;
        emit<R>(out, tk, a.shape(), edge, opts, 0, cursor);
        ENSURES(cursor == tk.tok.size());
        return out;
      }
    }

    // Implementation shared by the std::formatter specializations for nda::Array and nda::array_with_opts.
    template <Array A>
    struct array_formatter {
      protected:
      // Component type of the elements, i.e. the type the format spec is applied to.
      using component_t = typename remove_complex<get_value_t<A>>::type;

      // The format spec of an element, split into its parts.
      elem_spec spec_{};

      // Read a dynamic width or precision argument from the format context.
      template <typename FmtCtx>
      static long dynamic_arg(FmtCtx &ctx, int id) {
        auto visitor = [](auto val) -> long {
          using V = decltype(val);
          if constexpr (std::is_integral_v<V> and not std::is_same_v<V, bool>) {
            if constexpr (std::is_signed_v<V>) {
              if (val < 0) throw std::format_error("Error in std::formatter<nda::Array>: Negative width or precision argument");
            }
            return static_cast<long>(val);
          } else {
            throw std::format_error("Error in std::formatter<nda::Array>: Width or precision argument is not an integer");
          }
        };
        auto arg = ctx.arg(static_cast<size_t>(id));
#if __cpp_lib_format >= 202306L
        return arg.visit(visitor);
#else
        return std::visit_format_arg(visitor, arg);
#endif
      }

      // Hand the component part of the spec to the formatter of the component type. Everything it rejects, e.g.
      // "{:q}" or a precision on an integral component, is thereby rejected at compile time, exactly as it is for a
      // plain component, without this formatter having to know the grammar of the spec.
      constexpr void validate_component_spec() const {
        auto buf    = std::array<char, 64>{};
        size_t n    = 0;
        auto append = [&buf, &n](std::string_view s) {
          if (n + s.size() > buf.size()) throw std::format_error("Error in std::formatter<nda::Array>: Format spec is too long");
          for (char c : s) buf[n++] = c;
        };
        append(spec_.head);
        if (spec_.prec_arg >= 0) append(".0");
        append(spec_.tail);

        auto const sv = std::string_view(buf.data(), n);
        auto pc       = std::format_parse_context(sv);
        auto f        = std::formatter<component_t, char>{};
        if (f.parse(pc) != sv.end()) throw std::format_error("Error in std::formatter<nda::Array>: Invalid format spec for the elements");
      }

      // Format an array with the given options into the output of the format context.
      template <typename FmtCtx>
      auto format_impl(A const &a, print_options const &opts, FmtCtx &ctx) const {
        // The format strings of the components and the padding of an element only depend on the spec, so they are
        // built once for the whole array.
        auto const width = spec_.width_arg >= 0 ? dynamic_arg(ctx, spec_.width_arg) : spec_.width;
        auto prec        = std::string{};
        if (spec_.prec_arg >= 0) prec = "." + std::to_string(dynamic_arg(ctx, spec_.prec_arg));

        // For anything but a complex element the '0' flag asks for sign aware zero padding, which only the formatter
        // of the component can do, so it is passed on to it together with the width. For a complex element the zeros
        // pad the whole "(re+imi)" instead, as they do in fmtlib. An explicit alignment makes the flag ignored, as it
        // does in std::format.
        bool const zero      = spec_.zero and spec_.align == 0 and width > 0;
        bool const zero_comp = zero and not is_complex_v<get_value_t<A>>;
        auto const zero_w    = zero_comp ? "0" + std::to_string(width) : std::string{};

        auto const re_fmt = "{:" + std::string{spec_.head} + zero_w + prec + std::string{spec_.tail} + "}";
        auto const im_fmt = "{:" + imag_head(spec_.head) + zero_w + prec + std::string{spec_.tail} + "}";

        // Padding the element as a string leaves the standard library in charge of the estimated width. The default
        // alignment is right, consistent with the way the columns of an array are aligned.
        auto pad_fmt = std::string{};
        if (width > 0 and not zero_comp) {
          pad_fmt = "{:";
          pad_fmt += zero ? '0' : spec_.fill;
          pad_fmt += zero ? '>' : (spec_.align ? spec_.align : '>');
          pad_fmt += std::to_string(width) + "}";
        }

        auto const s = render_array(a, opts, [&](auto const &x) {
          auto tok = render_element(x, re_fmt, im_fmt);
          return pad_fmt.empty() ? tok : format_value(tok, pad_fmt);
        });
        return std::copy(s.begin(), s.end(), ctx.out());
      }

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
          if (*it == '{') {
            // a dynamic width or precision argument, which cannot be nested
            while (it != ctx.end() and *it != '}') ++it;
            if (it == ctx.end()) throw std::format_error("Error in std::formatter<nda::Array>: Unterminated dynamic width or precision argument");
          }
          ++it;
        }
        spec_ = split_elem_spec(std::string_view(first, it), ctx);
        validate_component_spec();
        return it;
      }
    };

    // The type an argument of nda::format_with_opts is formatted as. Written as a specialization rather than with
    // std::conditional_t, which would form the constrained nda::array_with_opts type for every argument type.
    template <typename T, bool = Array<std::remove_cvref_t<T>>>
    struct wrapped {
      using type = std::decay_t<T>;
    };

    template <typename T>
    struct wrapped<T, true> {
      using type = array_with_opts<std::remove_cvref_t<T>>;
    };

    template <typename T>
    using wrapped_t = typename wrapped<T>::type;

    // Bundle an nda::Array argument of nda::format_with_opts with the options, pass everything else through.
    template <typename T>
    wrapped_t<T> wrap_with_opts(T &&x, print_options const &opts) {
      if constexpr (Array<std::remove_cvref_t<T>>)
        return {x, opts};
      else
        return std::forward<T>(x);
    }

  } // namespace detail

  /**
   * @brief An nda::Array bundled with the nda::print_options it is formatted with.
   *
   * @details Created with nda::with_opts, e.g. `std::format("{}", nda::with_opts(a, {.edgeitems = 1}))`.
   *
   * @tparam A nda::Array type.
   */
  template <Array A>
  struct array_with_opts {
    /// nda::Array to be formatted.
    A const &a; // NOLINT (a reference is what we want here: this is a short lived proxy passed straight to std::format)

    /// nda::print_options to format it with.
    print_options opts;
  };

  /**
   * @brief Bundle an nda::Array with the nda::print_options it should be formatted with.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @param opts nda::print_options.
   * @return nda::array_with_opts object, formattable with `std::format` and `std::format_to`.
   */
  template <Array A>
  array_with_opts<A> with_opts(A const &a, print_options const &opts) {
    return {a, opts};
  }

  /**
   * @brief Generalization of `std::format` which formats every nda::Array argument with the given options.
   *
   * @details Arguments which are not nda::Array types are formatted as usual, e.g.
   * `nda::format_with_opts({.threshold = -1}, "{}: {}", name, a)`.
   *
   * @tparam Ts Types of the format arguments.
   * @param opts nda::print_options applied to every nda::Array argument.
   * @param fmt Format string.
   * @param ts Format arguments.
   * @return `std::string` containing the formatted output.
   */
  template <typename... Ts>
  std::string format_with_opts(print_options const &opts, std::format_string<detail::wrapped_t<Ts>...> fmt, Ts &&...ts) {
    auto args = std::tuple<detail::wrapped_t<Ts>...>{detail::wrap_with_opts(std::forward<Ts>(ts), opts)...};
    return std::apply([&fmt](auto &...xs) { return std::vformat(fmt.get(), std::make_format_args(xs...)); }, args);
  }

  /** @} */

} // namespace nda

/**
 * @ingroup av_utils
 * @brief `std::formatter` specialization for nda::Array types.
 *
 * @details Formats the values of the array element-wise, in a rank-generic, numpy-like format:
 *
 * ```
 * [[ 1  2  3]
 *  [10 20 30]]
 * ```
 *
 * Arrays with more than nda::print_options::threshold elements are summarized with `...`, keeping
 * nda::print_options::edgeitems elements at both ends of every truncated dimension. This formatter always uses
 * default constructed options, use nda::with_opts or nda::format_with_opts for other ones.
 *
 * The format spec is applied to the *elements*, i.e. `std::format("{:.3f}", a)` applies `.3f` to every element of
 * `a`. Its fill, align and width apply to the whole element, everything else to the element itself or, for a
 * `std::complex` element, to its real and imaginary part. Since the columns of an array are padded to their common
 * width anyway, a width in the spec acts as a minimum column width. The '0' flag keeps its usual sign aware meaning
 * and is therefore applied to the element itself, except for a `std::complex` element, where it pads the whole
 * `(re+imi)` as fmtlib does. Dynamic width and precision arguments, e.g. `std::format("{:.{}f}", a, 3)`, are
 * supported.
 *
 * `std::complex` elements are formatted like fmtlib does, as `(re+imi)`, except that the parentheses are always
 * there. The imaginary part always carries a sign, a user supplied one is not doubled, and a space precedes the `i`
 * if it is not finite, e.g. `(1+inf i)`.
 *
 * Note that this also formats the lazy expression types, in which case the values of the expression are printed
 * rather than its structure. This differs from nda::operator<<, which prints the structure of an expression.
 *
 * Only `char` output is supported, i.e. formatting an array into a `std::wstring` is a compile time error.
 *
 * @tparam A nda::Array type to be formatted.
 */
template <nda::Array A>
  requires(nda::detail::has_std_formatter_v<typename nda::remove_complex<nda::get_value_t<A>>::type>)
struct std::formatter<A, char> : nda::detail::array_formatter<A> {
  /**
   * @brief Format an nda::Array into the output of the format context, using default nda::print_options.
   *
   * @param a nda::Array object to format.
   * @param ctx Format context to write the output to.
   * @return Iterator past the end of the written output.
   */
  template <typename FmtCtx>
  auto format(A const &a, FmtCtx &ctx) const {
    return this->format_impl(a, nda::print_options{}, ctx);
  }
};

/**
 * @ingroup av_utils
 * @brief `std::formatter` specialization for nda::array_with_opts, i.e. for nda::Array types formatted with
 * non-default nda::print_options.
 *
 * @details Identical to the `std::formatter` specialization for nda::Array types, except that the options carried by
 * the argument are used instead of default constructed ones.
 *
 * @tparam A nda::Array type to be formatted.
 */
template <nda::Array A>
  requires(nda::detail::has_std_formatter_v<typename nda::remove_complex<nda::get_value_t<A>>::type>)
struct std::formatter<nda::array_with_opts<A>, char> : nda::detail::array_formatter<A> {
  /**
   * @brief Format an nda::array_with_opts into the output of the format context.
   *
   * @param x nda::array_with_opts object to format.
   * @param ctx Format context to write the output to.
   * @return Iterator past the end of the written output.
   */
  template <typename FmtCtx>
  auto format(nda::array_with_opts<A> const &x, FmtCtx &ctx) const {
    return this->format_impl(x.a, x.opts, ctx);
  }
};

#endif // __cpp_lib_format

#endif // STDUTILS_FORMAT_H
