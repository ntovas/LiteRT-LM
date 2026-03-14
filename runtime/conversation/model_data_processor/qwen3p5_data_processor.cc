// Copyright 2026 The ODML Authors.
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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

bool PrefaceHasTools(const std::optional<Preface>& preface) {
  return preface.has_value() && std::holds_alternative<JsonPreface>(*preface) &&
         !std::get<JsonPreface>(*preface).tools.empty();
}

SyntaxType ToSyntaxType(Qwen3p5ToolCallSyntax syntax) {
  switch (syntax) {
    case Qwen3p5ToolCallSyntax::kQwen3p5Xml:
      return SyntaxType::kQwen3p5Xml;
  }
  return SyntaxType::kQwen3p5Xml;
}

std::string JoinTextParts(const json& content) {
  if (!content.is_array()) {
    return "";
  }
  std::string result;
  for (const auto& item : content) {
    if (item.is_object() && item.contains("text") && item["text"].is_string()) {
      result += item["text"].get<std::string>();
    }
  }
  return result;
}

void ExtractReasoningContent(absl::string_view response_text,
                             std::string* reasoning_content,
                             std::string* content) {
  constexpr absl::string_view kThinkStart = "<think>";
  constexpr absl::string_view kThinkEnd = "</think>";
  const size_t think_start = response_text.find(kThinkStart);
  if (think_start == absl::string_view::npos) {
    *content = std::string(response_text);
    return;
  }
  const size_t think_content_start = think_start + kThinkStart.size();
  const size_t think_end = response_text.find(kThinkEnd, think_content_start);
  if (think_end == absl::string_view::npos) {
    *content = std::string(response_text);
    return;
  }
  *reasoning_content = std::string(
      response_text.substr(think_content_start, think_end - think_content_start));
  *content = std::string(response_text.substr(0, think_start));
  content->append(response_text.substr(think_end + kThinkEnd.size()));
}

json MakeTextContent(absl::string_view text) {
  return json::array({{{"type", "text"}, {"text", std::string(text)}}});
}

}  // namespace

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig config,
                             std::optional<Preface> preface) {
  return absl::WrapUnique(
      new Qwen3p5DataProcessor(std::move(config), std::move(preface)));
}

absl::StatusOr<json> Qwen3p5DataProcessor::MessageToTemplateInput(
    const json& message) const {
  json result = message;
  const std::string role =
      result.contains("role") && result["role"].is_string()
          ? result["role"].get<std::string>()
          : "";
  if (result.contains("content")) {
    if (role == "tool" && !result["content"].is_string()) {
      result["content"] = result["content"].dump();
    } else if (result["content"].is_array()) {
      result["content"] = JoinTextParts(result["content"]);
    }
  }

  if (role == "assistant") {
    if (result.contains("reasoning_content") &&
        result["reasoning_content"].is_array()) {
      result["reasoning_content"] = JoinTextParts(result["reasoning_content"]);
    }
    if ((!result.contains("reasoning_content") ||
         !result["reasoning_content"].is_string()) &&
        result.contains("content") && result["content"].is_string()) {
      std::string reasoning_content;
      std::string content;
      ExtractReasoningContent(result["content"].get<std::string>(),
                              &reasoning_content, &content);
      if (!reasoning_content.empty()) {
        result["reasoning_content"] = reasoning_content;
        result["content"] = content;
      }
    }
  }

  return result;
}

absl::StatusOr<std::vector<InputData>>
Qwen3p5DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const json& messages,
    const Qwen3p5DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  input_data.emplace_back(InputText(rendered_template_prompt));
  return input_data;
}

absl::StatusOr<Message> Qwen3p5DataProcessor::ToMessageImpl(
    const Responses& responses, const Qwen3p5DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  std::string reasoning_content;
  std::string content_without_thinking;
  ExtractReasoningContent(response_text, &reasoning_content,
                          &content_without_thinking);

  json message = {{"role", "assistant"}};
  if (PrefaceHasTools(preface_)) {
    ASSIGN_OR_RETURN(
        json content_and_tool_calls,
        ParseTextAndToolCalls(content_without_thinking, config_.code_fence_start,
                              config_.code_fence_end,
                              ToSyntaxType(config_.tool_call_syntax),
                              config_.escape_fence_strings,
                              config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = MakeTextContent(content_without_thinking);
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
