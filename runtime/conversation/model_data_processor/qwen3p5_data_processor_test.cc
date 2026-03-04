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

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

constexpr char kImageTestdataDir[] =
    "litert_lm/runtime/components/preprocessor/testdata/";

std::string ReadFile(absl::string_view path) {
  std::ifstream ifstr(std::string(path), std::ios::binary);
  std::stringstream contents;
  contents << ifstr.rdbuf();
  return contents.str();
}

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

MATCHER(IsInputImage, "") {
  if (!std::holds_alternative<InputImage>(arg)) {
    return false;
  }
  const auto& img = std::get<InputImage>(arg);
  // Patchify produces a TensorBufferMap; fixed-resize produces a TensorBuffer.
  return img.IsTensorBuffer() || img.IsTensorBufferMap();
}

TEST(Qwen3p5DataProcessorTest, ToInputDataVector_TextOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  const std::string rendered_template_prompt =
      "<|im_start|>user\ntest prompt\n<|im_end|>\n<|im_start|>assistant\n";
  const nlohmann::ordered_json messages = json::array({
      {{"role", "user"}, {"content", "test prompt"}},
  });
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text(rendered_template_prompt);
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Qwen3p5DataProcessorTest, ToInputDataVector_ImageContent) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  const std::string image_path =
      (std::filesystem::path(::testing::SrcDir()) / kImageTestdataDir /
       "apple.png")
          .string();

  // Rendered template prompt contains the Qwen3.5 image placeholder.
  const std::string rendered_template_prompt =
      "<|im_start|>user\n"
      "<|vision_start|><|image_pad|><|vision_end|>"
      "Describe this image\n<|im_end|>\n<|im_start|>assistant\n";

  const nlohmann::ordered_json messages = json::array({
      {{"role", "user"},
       {"content", json::array({
           {{"type", "image"}, {"path", image_path}},
           {{"type", "text"}, {"text", "Describe this image"}},
       })}},
  });

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  // Expected: InputText(boi), InputImage(preprocessed), InputText(eoi),
  // InputText(remaining).
  InputText expected_boi("<|im_start|>user\n<|vision_start|>");
  InputText expected_eoi("<|vision_end|>");
  InputText expected_tail(
      "Describe this image\n<|im_end|>\n<|im_start|>assistant\n");
  EXPECT_THAT(input_data,
              ElementsAre(HasInputText(&expected_boi), IsInputImage(),
                          HasInputText(&expected_eoi),
                          HasInputText(&expected_tail)));
}

TEST(Qwen3p5DataProcessorTest,
     ToInputDataVector_VideoContent_ReturnsUnimplemented) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  const std::string rendered_template_prompt = "some prompt";
  const nlohmann::ordered_json messages = json::array({
      {{"role", "user"},
       {"content", json::array({
           {{"type", "video"}, {"path", "/some/video.mp4"}},
       })}},
  });
  auto result =
      processor->ToInputDataVector(rendered_template_prompt, messages, {});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(Qwen3p5DataProcessorTest,
     ToInputDataVector_VideoUrlContent_ReturnsUnimplemented) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  const std::string rendered_template_prompt = "some prompt";
  const nlohmann::ordered_json messages = json::array({
      {{"role", "user"},
       {"content", json::array({
           {{"type", "video_url"}, {"video_url", {{"url", "http://example.com/v.mp4"}}}},
       })}},
  });
  auto result =
      processor->ToInputDataVector(rendered_template_prompt, messages, {});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(Qwen3p5DataProcessorTest,
     ToInputDataVector_ImageCountMismatch_ReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  // Rendered prompt has an image placeholder but no image is in the messages.
  const std::string rendered_template_prompt =
      "<|vision_start|><|image_pad|><|vision_end|>";
  const nlohmann::ordered_json messages = json::array({
      {{"role", "user"}, {"content", "no image here"}},
  });
  auto result =
      processor->ToInputDataVector(rendered_template_prompt, messages, {});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Qwen3p5DataProcessorTest, ToMessage_NoTools) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(
      json_message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessage_WithThinking) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"<think>step by step reasoning</think>\nfinal answer"}),
          std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(json_message,
            json({{"role", "assistant"},
                  {"content", {{{"type", "text"}, {"text", "final answer"}}}},
                  {"reasoning_content", "step by step reasoning"}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessage_EmptyThinking) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"<think>\n\n</think>\nanswer text"}),
          std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  // Empty thinking_content should not be added to message.
  EXPECT_FALSE(json_message.contains("reasoning_content"));
  EXPECT_EQ(json_message["content"],
            json({{{"type", "text"}, {"text", "answer text"}}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessage_WithTools) {
  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::array();
  preface.tools.push_back(
      json{{"type", "function"}, {"function", {{"name", "func1"}}}});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"this is text and tool call "
                     "<tool_call>{\"name\":\"func1\",\"arguments\":{"
                     "\"arg1\":1}}</tool_call>"}),
          std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(json_message, nlohmann::ordered_json::parse(R"json({
              "role": "assistant",
                "content": [
                  {
                    "type": "text",
                    "text": "this is text and tool call "
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "func1",
                      "arguments": {
                        "arg1": 1
                      }
                    }
                  }
                ]
              })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessage_WithThinkingAndTools) {
  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::array();
  preface.tools.push_back(
      json{{"type", "function"}, {"function", {{"name", "func1"}}}});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"<think>let me think</think>\ntext "
                     "<tool_call>{\"name\":\"func1\",\"arguments\":{"
                     "\"arg1\":1}}</tool_call>"}),
          std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_TRUE(json_message.contains("reasoning_content"));
  EXPECT_EQ(json_message["reasoning_content"], "let me think");
  EXPECT_TRUE(json_message.contains("tool_calls"));
}

TEST(Qwen3p5DataProcessorTest, CodeFence) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}));
  EXPECT_EQ(processor->CodeFenceStart(), "<tool_call>");
  EXPECT_EQ(processor->CodeFenceEnd(), "</tool_call>");
}

}  // namespace
}  // namespace litert::lm
