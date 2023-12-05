<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 에이전트와 도구 [[agents-tools]]

<Tip warning={true}>

Transformers Agent는 실험 중인 API로 언제든지 변경될 수 있습니다. API 또는 기반 모델이 변경되기 쉽기 때문에 에이전트가 반환하는 결과도 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 알고 싶다면 [소개 가이드](../transformers_agents)를 읽어보세요. 이 페이지에는 기본 클래스에 대한 API 문서가 포함되어 있습니다.

## 에이전트 [[agents]]

에이전트에는 3개의 타입이 존재합니다. [`HfAgent`]는 오픈소스 모델을 위한 추론 엔드포인트를 사용하고, [`LocalAgent`]는 로컬에서 지정한 모델을 사용하며, [`OpenAiAgent`]는 OpenAI의 비공개 모델을 사용합니다.

### HfAgent [[hfagent]]

[[autodoc]] HfAgent

### LocalAgent [[localagent]]

[[autodoc]] LocalAgent

### OpenAiAgent [[openaiagent]]

[[autodoc]] OpenAiAgent

### AzureOpenAiAgent [[azureopenaiagent]]

[[autodoc]] AzureOpenAiAgent

### Agent [[agent]]

[[autodoc]] Agent
    - chat
    - run
    - prepare_for_new_chat

## 도구들 [[tools]]

### load_tool [[loadtool]]

[[autodoc]] load_tool

### Tool [[tool]]

[[autodoc]] Tool

### PipelineTool [[pipelinetool]]

[[autodoc]] PipelineTool

### RemoteTool [[remotetool]]

[[autodoc]] RemoteTool

### launch_gradio_demo [[launchgradiodemo]]

[[autodoc]] launch_gradio_demo

## 에이전트 타입 [[agent-types]]

에이전트의 도구들은 텍스트, 이미지, 오디오, 비디오 등 다양한 형식의 데이터를 서로 주고받고 처리할 수 있습니다. 이러한 다양한 데이터 타입 간의 호환성을 높이고 ipython 환경(jupyter, colab, ipython 노트북, ...)에서 데이터를 올바르게 표시하기 위해 래퍼 클래스를 구현합니다. 

래핑된 객체들은 초기의 기능을 그대로 유지해야 합니다. 예를 들어, 텍스트 객체는 여전히 문자열로, 이미지 객체는 `PIL.Image`로 작동해야 합니다.

이러한 타입에는 세 가지의 구체적인 목적이 있습니다:

- `to_raw` 함수를 사용하면 객체의 기본 형태가 반환됩니다.
- `to_string` 함수를 사용하면 해당 객체를 문자열로 반환합니다. 이는 `AgentText` 객체의 경우 그대로 문자열이, 다른 객체의 경우에는 직렬화된 파일 경로가 될 수 있습니다.
- ipython 커널에서 객체를 표시하면, 해당 객체가 적절하게 보여집니다.

### AgentText [[agenttext]]

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage [[agentimage]]

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio [[agentaudio]]

[[autodoc]] transformers.tools.agent_types.AgentAudio
