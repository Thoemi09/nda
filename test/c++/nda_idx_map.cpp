// Copyright (c) 2018-2020 Simons Foundation
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

#define NDA_ENFORCE_BOUNDCHECK
#include <gtest/gtest.h> // NOLINT

#include <nda/nda.hpp>

nda::range _;
nda::ellipsis ___;
using nda::slice_static::slice_stride_order;

using namespace nda;

template <typename... INT>
std::array<long, sizeof...(INT)> ma(INT... i) {
  return {i...};
}

//-----------------------

TEST(idxstat, Construct) { // NOLINT

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> i1{{1, 2, 3}};

  std::cerr << i1 << std::endl;
  EXPECT_TRUE(i1.lengths() == (ma(1, 2, 3))); //NOLINT
  EXPECT_TRUE(i1.strides() == (ma(6, 3, 1))); //NOLINT
}

//-----------------------

TEST(idxstat, eval) { // NOLINT

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> i1{{2, 7, 3}};
  EXPECT_TRUE(i1.strides() == (ma(21, 3, 1))); //NOLINT

  EXPECT_EQ(i1(1, 3, 2), 21 * 1 + 3 * 3 + 2 * 1); //NOLINT
}

//-------------------------

TEST(idxstat, to_idx) {

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> iC{{2, 7, 3}};
  idx_map<3, 0, Fortran_stride_order<3>, layout_prop_e::none> iF{{2, 7, 3}};
  auto iP = iF.transpose<encode(std::array{0, 2, 1})>();

  for (auto idx0 : range(2)) {
    auto iPv = iP.slice(idx0, _, _).second;

    for (auto idx1 : range(7)) {
      for (auto idx2 : range(3)) {
        EXPECT_TRUE(iPv.to_idx(iPv(idx2, idx1)) == ma(idx2, idx1));
        EXPECT_TRUE(iC.to_idx(iC(idx0, idx1, idx2)) == ma(idx0, idx1, idx2));
        EXPECT_TRUE(iF.to_idx(iF(idx0, idx1, idx2)) == ma(idx0, idx1, idx2));
        EXPECT_TRUE(iP.to_idx(iP(idx0, idx2, idx1)) == ma(idx0, idx2, idx1));
      }
    }
  }
}

//-----------------------

//TEST(idxstat, boundcheck) { // NOLINT

//idx_map<3, 0, layout_prop_e::none> i1{{2, 7, 3}};
////i1(21, 3, 18);
////EXPECT_THROW(i1(21, 3, 18), std::exception); //NOLINT
//}

//-----------------------

TEST(idxstat, slicemat) { // NOLINT

  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> i1{{10,10}};

  auto [offset2, i2] = slice_stride_order(i1, range(0,2), 2);

  static_assert(decltype(i2)::layout_prop == layout_prop_e::strided_1d, "000");

}

//-----------------------

TEST(idxstat, slice) { // NOLINT

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> i1{{1, 2, 3}};

  auto [offset2, i2] = slice_stride_order(i1, 0, _, 2);

  idx_map<1, 0, C_stride_order<1>, layout_prop_e::strided_1d> c2{{2}, {3}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2); //NOLINT
  EXPECT_EQ(offset2, 2); //NOLINT

  auto [offset3, i3] = slice_stride_order(i1, _, _, _);
  EXPECT_TRUE(i3 == i1); //NOLINT
  EXPECT_EQ(offset3, 0); //NOLINT
}

//-----------------------

TEST(idxstat, ellipsis) { // NOLINT

  EXPECT_EQ(16, encode(nda::slice_static::sliced_mem_stride_order(std::array<int, 3>{0, 1, 2}, std::array<int, 2>{1, 2})));

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> i1{{1, 2, 3}};
  auto [offset2, i2] = slice_stride_order(i1, 0, ___);

  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> c2{{2, 3}, {3, 1}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2); //NOLINT
  EXPECT_EQ(offset2, 0); //NOLINT

  auto [offset3, i3] = slice_stride_order(i1, ___);
  EXPECT_TRUE(i3 == i1); //NOLINT
  EXPECT_EQ(offset3, 0); //NOLINT
}

//-----------------------

TEST(idxstat, ellipsis2) { // NOLINT

  idx_map<5, 0, C_stride_order<5>, layout_prop_e::none> i1{{1, 2, 3, 4, 5}};
  std::cerr << i1 << std::endl;

  auto [offset2, i2] = slice_stride_order(i1, 0, ___, 3, 2);
  idx_map<2, 0, C_stride_order<2>, layout_prop_e::none> c2{{2, 3}, {60, 20}};

  std::cerr << i2 << std::endl;
  std::cerr << c2 << std::endl;
  EXPECT_TRUE(i2 == c2);                 //NOLINT
  EXPECT_EQ(offset2, i1(0, 0, 0, 3, 2)); //NOLINT
}

//----------- Iterator ------------

TEST(idxstat, iteratorC) { // NOLINT

  idx_map<3, 0, C_stride_order<3>, layout_prop_e::none> i1{{2, 3, 4}};

  std::cerr << i1 << std::endl;

  //int pos = 2;
  //for (auto p : i1) { // FIXME To be implemented
  //EXPECT_EQ(p, pos++); //NOLINT
  //}

  //pos = 0;
  //for (auto [p, i] : enumerate_indices(i1)) {
  //EXPECT_EQ(p, pos++); //NOLINT
  //std::cerr << i << std::endl;
  //}
}

TEST(idxstat, for_each) { // NOLINT

  {
    std::stringstream fs;
    auto l = [&fs](int i, int j, int k) { fs << i << j << k << " "; };

    for_each(std::array<long, 3>{1, 2, 3}, l);
    EXPECT_EQ(fs.str(), "000 001 002 010 011 012 ");
  }

  //{
  //std::stringstream fs;
  //auto l = [&fs](int i, int j, int k) { fs << i << j << k << " "; };

  //for_each(std::array<long, 3>{1, 2, 3}, l, traversal::Fortran);
  //EXPECT_EQ(fs.str(), "000 010 001 011 002 012 ");
  //}
}
