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

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"     // from @com_google_absl
#include "absl/status/statusor.h"   // from @com_google_absl
#include "absl/types/span.h"        // from @com_google_absl
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

absl::StatusOr<MRoPEResult> ComputeQwen35MRoPEPositions(
    absl::Span<const int32_t> token_ids,
    const std::vector<std::array<int, 3>>& image_grid_thw_list,
    int spatial_merge_size, int batch_size, int seq_len) {
  MRoPEResult result;
  result.position_ids.resize(3 * batch_size * seq_len, 0);
  result.rope_deltas.resize(batch_size, 0);

  for (int b = 0; b < batch_size; ++b) {
    const int32_t* ids = token_ids.data() + b * seq_len;
    int current_pos = 0;
    int image_idx = 0;
    int t = 0;

    while (t < seq_len) {
      if (ids[t] != ExecutorVisionData::kSpecialToken) {
        // Text token: all three RoPE dims get the same sequential position.
        result.position_ids[0 * batch_size * seq_len + b * seq_len + t] =
            current_pos;
        result.position_ids[1 * batch_size * seq_len + b * seq_len + t] =
            current_pos;
        result.position_ids[2 * batch_size * seq_len + b * seq_len + t] =
            current_pos;
        ++current_pos;
        ++t;
      } else {
        // Vision run: consume all consecutive special tokens.
        int run_start = t;
        while (t < seq_len &&
               ids[t] == ExecutorVisionData::kSpecialToken) {
          ++t;
        }

        if (image_idx >= static_cast<int>(image_grid_thw_list.size())) {
          return absl::InvalidArgumentError(
              "More vision token runs than images in image_grid_thw_list.");
        }
        const auto& thw = image_grid_thw_list[image_idx];
        const int llm_grid_t = thw[0];
        const int llm_grid_h = thw[1] / spatial_merge_size;
        const int llm_grid_w = thw[2] / spatial_merge_size;

        // Fill positions in row-major order (ti, hi, wi).
        int patch_idx = run_start;
        for (int ti = 0; ti < llm_grid_t; ++ti) {
          for (int hi = 0; hi < llm_grid_h; ++hi) {
            for (int wi = 0; wi < llm_grid_w; ++wi) {
              result.position_ids[0 * batch_size * seq_len + b * seq_len +
                                  patch_idx] = ti + current_pos;
              result.position_ids[1 * batch_size * seq_len + b * seq_len +
                                  patch_idx] = hi + current_pos;
              result.position_ids[2 * batch_size * seq_len + b * seq_len +
                                  patch_idx] = wi + current_pos;
              ++patch_idx;
            }
          }
        }

        current_pos += std::max(llm_grid_h, llm_grid_w);
        ++image_idx;
      }
    }

    // rope_delta = max_position + 1 - seq_len = current_pos - seq_len.
    result.rope_deltas[b] = current_pos - seq_len;
  }

  return result;
}

std::vector<int32_t> ComputeQwen35DecodePosition(
    absl::Span<const int32_t> rope_deltas, int step, int batch_size) {
  std::vector<int32_t> positions(3 * batch_size);
  for (int dim = 0; dim < 3; ++dim) {
    for (int b = 0; b < batch_size; ++b) {
      positions[dim * batch_size + b] = step + rope_deltas[b];
    }
  }
  return positions;
}

}  // namespace litert::lm
