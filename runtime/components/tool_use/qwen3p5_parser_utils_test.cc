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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;

TEST(Qwen3p5ParserUtilsTest, ParseSingleToolCall) {
  EXPECT_THAT(
      ParseQwen3p5ToolCalls(
          "<function=get_weather><parameter=location>\"Paris\"</parameter>"
          "</function>"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([
        {
          "name": "get_weather",
          "arguments": {
            "location": "Paris"
          }
        }
      ])json")));
}

TEST(Qwen3p5ParserUtilsTest, ParseMultipleToolCalls) {
  EXPECT_THAT(
      ParseQwen3p5ToolCalls(
          "<function=get_weather><parameter=location>\"Paris\"</parameter>"
          "</function>"
          "<function=get_time><parameter=timezone>\"UTC\"</parameter>"
          "</function>"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([
        {
          "name": "get_weather",
          "arguments": {
            "location": "Paris"
          }
        },
        {
          "name": "get_time",
          "arguments": {
            "timezone": "UTC"
          }
        }
      ])json")));
}

TEST(Qwen3p5ParserUtilsTest, ParseJsonAndRawStringArguments) {
  EXPECT_THAT(
      ParseQwen3p5ToolCalls(
          "<function=search><parameter=count>3</parameter>"
          "<parameter=filters>{\"lang\":\"en\"}</parameter>"
          "<parameter=query>weather in paris</parameter></function>"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([
        {
          "name": "search",
          "arguments": {
            "count": 3,
            "filters": {
              "lang": "en"
            },
            "query": "weather in paris"
          }
        }
      ])json")));
}

TEST(Qwen3p5ParserUtilsTest, PreservesWhitespaceInQuotedAndRawStrings) {
  EXPECT_THAT(
      ParseQwen3p5ToolCalls(
          "<function=search>"
          "<parameter=quoted>\"  padded  \"</parameter>"
          "<parameter=raw>  keep me  </parameter>"
          "</function>"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([
        {
          "name": "search",
          "arguments": {
            "quoted": "  padded  ",
            "raw": "  keep me  "
          }
        }
      ])json")));
}

TEST(Qwen3p5ParserUtilsTest, InvalidInputFails) {
  EXPECT_FALSE(
      ParseQwen3p5ToolCalls("<function=get_weather><parameter=location>")
          .ok());
}

}  // namespace
}  // namespace litert::lm
