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

#include "runtime/components/tool_use/qwen3p5_parser_utils.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kFunctionStart = "<function=";
constexpr absl::string_view kFunctionEnd = "</function>";
constexpr absl::string_view kParameterStart = "<parameter=";
constexpr absl::string_view kParameterEnd = "</parameter>";

void SkipWhitespace(absl::string_view text, size_t* cursor) {
  while (*cursor < text.size() &&
         absl::ascii_isspace(static_cast<unsigned char>(text[*cursor]))) {
    ++(*cursor);
  }
}

absl::StatusOr<absl::string_view> ParseTagName(absl::string_view text,
                                               absl::string_view prefix,
                                               size_t* cursor) {
  if (!absl::StartsWith(text.substr(*cursor), prefix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected tag prefix: ", prefix));
  }
  const size_t name_start = *cursor + prefix.size();
  const size_t name_end = text.find('>', name_start);
  if (name_end == absl::string_view::npos) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing closing '>' for tag prefix: ", prefix));
  }
  *cursor = name_end + 1;
  return text.substr(name_start, name_end - name_start);
}

absl::StatusOr<nlohmann::ordered_json> ParseParameterValue(
    absl::string_view raw_value) {
  auto parsed =
      nlohmann::ordered_json::parse(std::string(raw_value), nullptr, false);
  if (!parsed.is_discarded()) {
    return parsed;
  }
  const std::string trimmed = std::string(absl::StripAsciiWhitespace(raw_value));
  if (trimmed != raw_value) {
    parsed = nlohmann::ordered_json::parse(trimmed, nullptr, false);
    if (!parsed.is_discarded()) {
      return parsed;
    }
  }
  return nlohmann::ordered_json(std::string(raw_value));
}

absl::StatusOr<nlohmann::ordered_json> ParseSingleFunction(absl::string_view text,
                                                           size_t* cursor) {
  ASSIGN_OR_RETURN(absl::string_view function_name,
                   ParseTagName(text, kFunctionStart, cursor));
  nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
  while (true) {
    SkipWhitespace(text, cursor);
    if (absl::StartsWith(text.substr(*cursor), kFunctionEnd)) {
      *cursor += kFunctionEnd.size();
      return nlohmann::ordered_json{{"name", std::string(function_name)},
                                    {"arguments", arguments}};
    }
    ASSIGN_OR_RETURN(absl::string_view parameter_name,
                     ParseTagName(text, kParameterStart, cursor));
    const size_t value_start = *cursor;
    const size_t value_end = text.find(kParameterEnd, value_start);
    if (value_end == absl::string_view::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing closing tag for parameter: ", parameter_name));
    }
    ASSIGN_OR_RETURN(arguments[std::string(parameter_name)],
                     ParseParameterValue(text.substr(value_start,
                                                    value_end - value_start)));
    *cursor = value_end + kParameterEnd.size();
  }
}

}  // namespace

absl::StatusOr<nlohmann::ordered_json> ParseQwen3p5ToolCalls(
    absl::string_view text) {
  nlohmann::ordered_json tool_calls = nlohmann::ordered_json::array();
  size_t cursor = 0;
  while (true) {
    SkipWhitespace(text, &cursor);
    if (cursor >= text.size()) {
      break;
    }
    ASSIGN_OR_RETURN(auto tool_call, ParseSingleFunction(text, &cursor));
    tool_calls.push_back(tool_call);
  }
  if (tool_calls.empty()) {
    return absl::InvalidArgumentError("No Qwen3.5 tool calls found.");
  }
  return tool_calls;
}

}  // namespace litert::lm
