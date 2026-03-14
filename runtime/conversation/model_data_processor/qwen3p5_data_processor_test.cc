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

#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  if (!text_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == text_input->GetRawTextString().value();
}

TEST(Qwen3p5DataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector("<|im_start|>user\nhi<|im_end|>\n", {}, {}));
  InputText expected_text("<|im_start|>user\nhi<|im_end|>\n");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Qwen3p5DataProcessorTest, ToMessageParsesThinking) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing,
                                     {"<think>\nreasoning\n</think>\n\nanswer"}),
                           std::monostate{}));

  const json& json_message = std::get<json>(message);
  EXPECT_EQ(json_message, json::parse(R"json({
              "role": "assistant",
              "content": [
                {
                  "type": "text",
                  "text": "\n\nanswer"
                }
              ],
              "reasoning_content": "\nreasoning\n"
            })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageParsesXmlToolCalls) {
  JsonPreface preface;
  preface.tools = json::array(
      {json{{"type", "function"}, {"function", {{"name", "get_weather"}}}}});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"<think>reason</think>Let me check.\n<tool_call>"
                     "<function=get_weather>"
                     "<parameter=location>\"Paris\"</parameter>"
                     "</function></tool_call>"}),
          std::monostate{}));

  const json& json_message = std::get<json>(message);
  EXPECT_EQ(json_message, json::parse(R"json({
              "role": "assistant",
              "content": [
                {
                  "type": "text",
                  "text": "Let me check.\n"
                }
              ],
              "tool_calls": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_weather",
                    "arguments": {
                      "location": "Paris"
                    }
                  }
                }
              ],
              "reasoning_content": "reason"
            })json"));
}

TEST(Qwen3p5DataProcessorTest, MessageToTemplateInputNormalizesAssistantAndTool) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const json assistant_message,
      processor->MessageToTemplateInput(json::parse(R"json({
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "<think>reason</think>answer"
          }
        ]
      })json")));
  EXPECT_EQ(assistant_message, json::parse(R"json({
              "role": "assistant",
              "content": "answer",
              "reasoning_content": "reason"
            })json"));

  ASSERT_OK_AND_ASSIGN(
      const json tool_message,
      processor->MessageToTemplateInput(json::parse(R"json({
        "role": "tool",
        "content": {
          "temperature": 72
        }
      })json")));
  EXPECT_EQ(tool_message["content"], "{\"temperature\":72}");

  ASSERT_OK_AND_ASSIGN(
      const json tool_array_message,
      processor->MessageToTemplateInput(json::parse(R"json({
        "role": "tool",
        "content": [
          {
            "type": "text",
            "text": "value"
          }
        ]
      })json")));
  EXPECT_EQ(tool_array_message["content"],
            "[{\"type\":\"text\",\"text\":\"value\"}]");
}

}  // namespace
}  // namespace litert::lm
