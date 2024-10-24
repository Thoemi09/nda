// Copyright (c) 2019-2020 Simons Foundation
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

#include <nda/nda.hpp>
#include <gtest/gtest.h>

#if defined(__has_feature)
#if !__has_feature(address_sanitizer)

TEST(Array, BadAlloc) { //NOLINT
  EXPECT_THROW(nda::vector<int>(long(1e16)), std::bad_alloc);
}

#else

TEST(Array, ASANPoison) { //NOLINT
  long *p;
  {
    nda::array<long, 2> A(3, 3);
    A() = 3;
    p   = &(A(0, 0));
  }

  EXPECT_EQ(__asan_address_is_poisoned(p), 1);
}

#endif
#endif
