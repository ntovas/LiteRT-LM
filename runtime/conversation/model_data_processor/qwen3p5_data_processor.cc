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

#include "runtime/conversation/model_data_processor/qwen3p5_data_processor.h"

#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig config,
                              std::optional<Preface> preface) {
  return absl::WrapUnique(
      new Qwen3p5DataProcessor(std::move(config), std::move(preface),
                               std::make_unique<StbImagePreprocessor>()));
}

absl::StatusOr<nlohmann::ordered_json>
Qwen3p5DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_array()) {
    const auto& content = message["content"];
    if (content.size() == 1 && content[0].contains("text")) {
      auto result = nlohmann::ordered_json::object(
          {{"role", message["role"]}, {"content", content[0]["text"]}});
      return result;
    }
  }
  return message;
}

absl::StatusOr<std::vector<InputData>>
Qwen3p5DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Qwen3p5DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;

  // Pre-pass: collect images, reject video.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        if (!item.contains("type")) {
          continue;
        }
        const std::string& type = item["type"].get_ref<const std::string&>();
        if (type == "video" || type == "video_url") {
          return absl::UnimplementedError(
              "Qwen3.5 video input is not supported.");
        }
        if (type == "image") {
          ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                           LoadItemData(item));
          image_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  // Set up image preprocessing parameters (dynamic resolution via patchify).
  // patch_size=16 is fixed by the Qwen3.5 ViT architecture.
  // max_num_patches=INT_MAX means no artificial cap: the model's own
  // max_pixels constraint (16,777,216 px) naturally bounds memory usage.
  ImagePreprocessParameter image_params;
  image_params.SetPatchifyConfig(
      {.patch_width = 16,
       .patch_height = 16,
       .max_num_patches = std::numeric_limits<int>::max()});

  // Replace image placeholders with preprocessed image data.
  RE2 re_delimiter(
      R"((<\|vision_start\|><\|image_pad\|><\|vision_end\|>))");
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_before(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    input_data.emplace_back(
        InputText(absl::StrCat(text_before, config_.boi_token)));
    if (image_files.empty()) {
      return absl::InvalidArgumentError(
          "Provided less images than expected in the prompt.");
    }
    auto image_file = std::move(image_files.front());
    image_files.pop_front();
    ASSIGN_OR_RETURN(
        auto preprocessed_image,
        image_preprocessor_->Preprocess(
            InputImage(std::string(
                static_cast<const char*>(image_file->data()),
                image_file->length())),
            image_params));
    input_data.emplace_back(InputImage(std::move(preprocessed_image)));
    input_data.emplace_back(InputText(config_.eoi_token));
  }

  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }

  // Add the remaining text in the prompt.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }

  return input_data;
}

absl::StatusOr<Message> Qwen3p5DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Qwen3p5DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json message = {{"role", "assistant"}};

  // Extract thinking tokens if present.
  std::string final_text;
  std::string reasoning_content;
  const std::string response_str(response_text);
  const auto think_start_pos =
      response_str.find(config_.thinking_start_token);
  const auto think_end_pos = response_str.find(config_.thinking_end_token);
  if (think_start_pos != std::string::npos &&
      think_end_pos != std::string::npos &&
      think_end_pos > think_start_pos) {
    const auto content_start =
        think_start_pos + config_.thinking_start_token.size();
    reasoning_content = response_str.substr(
        content_start, think_end_pos - content_start);
    // Strip leading/trailing whitespace so that <think>\n\n</think> is treated
    // as empty (no reasoning_content field added).
    const auto first_non_ws =
        reasoning_content.find_first_not_of("\n\r \t");
    if (first_non_ws == std::string::npos) {
      reasoning_content.clear();
    } else {
      const auto last_non_ws =
          reasoning_content.find_last_not_of("\n\r \t");
      reasoning_content =
          reasoning_content.substr(first_non_ws, last_non_ws - first_non_ws + 1);
    }
    const auto after_think =
        think_end_pos + config_.thinking_end_token.size();
    final_text = response_str.substr(after_think);
    // Strip leading newlines from the final text.
    const auto non_newline = final_text.find_first_not_of("\n\r");
    if (non_newline != std::string::npos) {
      final_text = final_text.substr(non_newline);
    } else {
      final_text.clear();
    }
  } else {
    final_text = response_str;
  }

  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        nlohmann::ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(final_text, config_.code_fence_start,
                              config_.code_fence_end, SyntaxType::kJson,
                              config_.escape_fence_strings,
                              config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", final_text}}});
  }

  if (!reasoning_content.empty()) {
    message["reasoning_content"] = reasoning_content;
  }

  return message;
}

absl::string_view Qwen3p5DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Qwen3p5DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
