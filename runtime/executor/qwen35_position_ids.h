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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QWEN35_POSITION_IDS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QWEN35_POSITION_IDS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"       // from @com_google_absl

namespace litert::lm {

// Result of a Qwen3.5 mRoPE position computation.
struct MRoPEResult {
  // Flattened [3 * batch_size * seq_len] in row-major order.
  // Index: dim * batch_size * seq_len + b * seq_len + t
  std::vector<int32_t> position_ids;
  // rope_delta per batch element, shape [batch_size].
  std::vector<int32_t> rope_deltas;
};

// Computes Qwen2-VL / Qwen3.5-VL 3D mRoPE positions for a single prefill.
//
// token_ids: flat [batch_size * seq_len]. Vision placeholders are -1
//   (ExecutorVisionData::kSpecialToken).
// image_grid_thw_list: per-image {T, H_patches, W_patches} in original patch
//   units. Images must appear in the same order as -1 runs in token_ids.
// spatial_merge_size: typically 2 (Qwen3.5 default).
// batch_size, seq_len: dimensions of token_ids.
//
// Returns MRoPEResult with position_ids shaped [3, batch_size, seq_len]
// (flattened row-major) and rope_deltas shaped [batch_size].
absl::StatusOr<MRoPEResult> ComputeQwen35MRoPEPositions(
    absl::Span<const int32_t> token_ids,
    const std::vector<std::array<int, 3>>& image_grid_thw_list,
    int spatial_merge_size, int batch_size, int seq_len);

// Computes the mRoPE position for a single decode step.
// Returns flat [3 * batch_size], indexed as dim * batch_size + b.
// Each element is step + rope_deltas[b], replicated across all 3 RoPE dims.
std::vector<int32_t> ComputeQwen35DecodePosition(
    absl::Span<const int32_t> rope_deltas, int step, int batch_size);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QWEN35_POSITION_IDS_H_
