// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/qwen35_position_ids.h"

#include <array>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "gtest/gtest.h"

namespace litert::lm {
namespace {

// kSpecialToken = -1 (from ExecutorVisionData).
constexpr int32_t kTok = 100;  // Any positive token id.
constexpr int32_t kVis = -1;   // Vision special token.

// Helper to make a flat position_ids vector from three dim-vectors.
std::vector<int32_t> MakeFlat(const std::vector<int32_t>& dim0,
                               const std::vector<int32_t>& dim1,
                               const std::vector<int32_t>& dim2) {
  std::vector<int32_t> out;
  out.insert(out.end(), dim0.begin(), dim0.end());
  out.insert(out.end(), dim1.begin(), dim1.end());
  out.insert(out.end(), dim2.begin(), dim2.end());
  return out;
}

// Test 1: text-only prefill → sequential positions, rope_delta=0.
TEST(Qwen35PositionIds, TextOnly) {
  constexpr int seq_len = 6;
  std::vector<int32_t> token_ids = {kTok, kTok, kTok, kTok, kTok, kTok};
  std::vector<std::array<int, 3>> grid_thw_list;  // empty

  auto result = ComputeQwen35MRoPEPositions(
      absl::MakeConstSpan(token_ids), grid_thw_list,
      /*spatial_merge_size=*/2, /*batch_size=*/1, seq_len);
  ASSERT_TRUE(result.ok()) << result.status();

  // All three dims should be [0, 1, 2, 3, 4, 5].
  std::vector<int32_t> seq = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(result->position_ids, MakeFlat(seq, seq, seq));
  EXPECT_EQ(result->rope_deltas, std::vector<int32_t>({0}));
}

// Test 2: one-image prefill.
// Sequence: 8 text tokens, then 4 vision special tokens (grid {1,4,4},
// spatial_merge_size=2 → llm_grid_h=2, llm_grid_w=2 → 4 patches).
TEST(Qwen35PositionIds, OneImage) {
  // grid {1, 4, 4} with spatial_merge_size=2 →
  //   llm_grid_t=1, llm_grid_h=2, llm_grid_w=2 → 1*2*2=4 vision tokens.
  constexpr int seq_len = 12;
  std::vector<int32_t> token_ids = {kTok, kTok, kTok, kTok, kTok, kTok, kTok,
                                    kTok, kVis, kVis, kVis, kVis};
  std::vector<std::array<int, 3>> grid_thw_list = {{{1, 4, 4}}};

  auto result = ComputeQwen35MRoPEPositions(
      absl::MakeConstSpan(token_ids), grid_thw_list,
      /*spatial_merge_size=*/2, /*batch_size=*/1, seq_len);
  ASSERT_TRUE(result.ok()) << result.status();

  // Text tokens 0–7: all dims = sequential 0..7.
  // Vision block (current_pos=8, llm_h=2, llm_w=2):
  //   (ti=0,hi=0,wi=0): t=8, h=8, w=8  → patch 8
  //   (ti=0,hi=0,wi=1): t=8, h=8, w=9  → patch 9
  //   (ti=0,hi=1,wi=0): t=8, h=9, w=8  → patch 10
  //   (ti=0,hi=1,wi=1): t=8, h=9, w=9  → patch 11
  //   current_pos += max(2,2) = 2 → current_pos=10
  // rope_delta = 10 - 12 = -2.
  std::vector<int32_t> dim0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8};
  std::vector<int32_t> dim1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9};
  std::vector<int32_t> dim2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9};
  EXPECT_EQ(result->position_ids, MakeFlat(dim0, dim1, dim2));
  EXPECT_EQ(result->rope_deltas, std::vector<int32_t>({-2}));
}

// Test 3: two-image prefill — verify image position blocks are sequentially
// separated by max(llm_grid_h, llm_grid_w) gaps.
TEST(Qwen35PositionIds, TwoImages) {
  // Sequence: 3 text, 4 vision (image1), 2 text, 4 vision (image2).
  // Both images have grid {1, 4, 4}, spatial_merge_size=2.
  constexpr int seq_len = 13;
  std::vector<int32_t> token_ids = {kTok, kTok, kTok,            // 0-2
                                    kVis, kVis, kVis, kVis,       // 3-6
                                    kTok, kTok,                    // 7-8
                                    kVis, kVis, kVis, kVis};      // 9-12
  std::vector<std::array<int, 3>> grid_thw_list = {{{1, 4, 4}}, {{1, 4, 4}}};

  auto result = ComputeQwen35MRoPEPositions(
      absl::MakeConstSpan(token_ids), grid_thw_list,
      /*spatial_merge_size=*/2, /*batch_size=*/1, seq_len);
  ASSERT_TRUE(result.ok()) << result.status();

  // Text 0-2: positions 0,1,2. current_pos=3.
  // Image1 (current_pos=3, llm_h=2, llm_w=2):
  //   patch3=(3,3,3), patch4=(3,3,4), patch5=(3,4,3), patch6=(3,4,4)
  //   current_pos = 3+max(2,2)=5.
  // Text 7-8: positions 5,6. current_pos=7.
  // Image2 (current_pos=7, llm_h=2, llm_w=2):
  //   patch9=(7,7,7), patch10=(7,7,8), patch11=(7,8,7), patch12=(7,8,8)
  //   current_pos = 7+max(2,2)=9.
  // rope_delta = 9 - 13 = -4.
  std::vector<int32_t> dim0 = {0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 7, 7, 7};
  std::vector<int32_t> dim1 = {0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8};
  std::vector<int32_t> dim2 = {0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8};
  EXPECT_EQ(result->position_ids, MakeFlat(dim0, dim1, dim2));
  EXPECT_EQ(result->rope_deltas, std::vector<int32_t>({-4}));

  // Verify the gap: image1 starts at t_pos=3, image2 at t_pos=7.
  // Difference = max(llm_grid_h=2, llm_grid_w=2) + 2 text tokens = 4.
  // This confirms images are separated by max(h,w) position advances.
}

// Test 4: decode position helper.
TEST(Qwen35PositionIds, DecodePosition) {
  std::vector<int32_t> rope_deltas = {5};
  auto positions = ComputeQwen35DecodePosition(
      absl::MakeConstSpan(rope_deltas), /*step=*/10, /*batch_size=*/1);
  EXPECT_EQ(positions, std::vector<int32_t>({15, 15, 15}));
}

// Test 5: decode position with negative rope_delta (typical for images).
TEST(Qwen35PositionIds, DecodePositionNegativeDelta) {
  std::vector<int32_t> rope_deltas = {-2};
  auto positions = ComputeQwen35DecodePosition(
      absl::MakeConstSpan(rope_deltas), /*step=*/14, /*batch_size=*/1);
  // step + delta = 14 + (-2) = 12. Replicated 3x.
  EXPECT_EQ(positions, std::vector<int32_t>({12, 12, 12}));
}

}  // namespace
}  // namespace litert::lm
