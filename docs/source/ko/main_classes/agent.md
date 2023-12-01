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

🤗 Transformers 에이전트는 실험적인 API로 언제든지 변경될 수 있습니다. 에이전트가 반환하는 결과는 API나 기반 모델이 변경될 수 있으므로 다를 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 알고 싶다면 [소개 가이드](../transformers_agents)를 읽어보세요. 이 페이지에는 기본 클래스에 대한 API 문서가 포함되어 있습니다.

## 에이전트 [[agents]]

우리는 세 가지 유형의 에이전트를 제공합니다: [`HfAgent`]는 오픈소스 모델을 위한 추론 엔드포인트를 사용하고, [`LocalAgent`]는 로컬에서 선택한 모델을 사용하며, [`OpenAiAgent`]는 OpenAI의 비공개 모델을 사용합니다.

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

에이전트는 도구 사이에서 다양한 유형의 객체를 처리할 수 있으며; 도구는 완전히 멀티모달하며 텍스트, 이미지, 오디오, 비디오 등 다른 유형을 수용하고 반환할 수 있습니다. 도구 간의 호환성을 높이고, ipython(jupyter, colab, ipython 노트북, ...)에서 이러한 반환값을 올바르게 렌더링하기 위해 우리는 이 유형들 주위에 래퍼 클래스를 구현합니다.

래핑된 객체는 처음과 같이 계속 동작해야 합니다; 텍스트 객체는 여전히 문자열처럼, 이미지 객체는 `PIL.Image`처럼 동작해야 합니다.

이 유형들은 세 가지 특정 목적을 가지고 있습니다:

- 유형에 `to_raw`를 호출하면 기본 객체를 반환해야 합니다
- 유형에 `to_string`을 호출하면 객체를 문자열로 반환해야 합니다: `AgentText`의 경우 문자열일 수 있지만, 다른 경우에는 객체의 직렬화된 버전의 경로일 것입니다
- ipython 커널에서 표시하면 객체가 올바르게 표시되어야 합니다

### AgentText [[agenttext]]

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage [[agentimage]]

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio [[agentaudio]]

[[autodoc]] transformers.tools.agent_types.AgentAudio
