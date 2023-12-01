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

# 텍스트 생성 전략 [[text-generation-strategies]]

텍스트 생성은 자유 형식의 텍스트 생성, 요약, 번역 등 많은 NLP 작업에서 중요합니다.
또한 음성을 텍스트로 변환하거나 이미지를 텍스트로 변환하는 등 텍스트를 출력으로 사용하는 다양한 혼합 모달리티 응용 프로그램에서도 역할을 수행하고 있습니다.
GPT2, XLNet, OpenAI GPT, CTRL, TransformerXL, XLM, Bart, T5, GIT, Whisper 등과 같은 다양한 모델이 텍스트를 생성할 수 있습니다.

다양한 작업을 위한 텍스트를 생성하기 위해 [`~transformers.generation_utils.GenerationMixin.generate`] 메소드를 사용하는 예시를 아래에서 확인할 수 있습니다:
* [텍스트 요약](./tasks/summarization#inference)
* [이미지 캡셔닝](./model_doc/git#transformers.GitForCausalLM.forward.example)
* [오디오 전사](./model_doc/whisper#transformers.WhisperForConditionalGeneration.forward.example)

generate 메소드에 대한 입력은 모델의 모달리티에 따라 다르며, AutoTokenizer 또는 AutoProcessor와 같은 모델의 전처리 클래스에서 반환됩니다. 모델의 전처리자가 여러 유형의 입력을 생성하는 경우, generate()에 모든 입력을 전달합니다. 각 모델의 전처리자에 대한 내용은 해당 모델의 문서에서 자세히 알아볼 수 있습니다.

텍스트를 생성하기 위해 출력 토큰을 선택하는 과정을 디코딩이라고 하며, `generate()` 메소드가 사용할 디코딩 전략을 사용자가 정의할 수 있습니다. 디코딩 전략을 수정해도 학습 중에 최적화되는 매개변수의 값은 변경되지 않습니다.
그러나, 디코딩 전략을 수정하면 생성된 텍스트 출력에 큰 영향을 미칠 수 있습니다. 텍스트의 반복을 줄이고 텍스트 생성을 더 일관성 있게 만들 수 있습니다.

이 가이드에서는 다음을 설명합니다:
* 기본 텍스트 생성 구성
* 일반적인 디코딩 전략과 주요 매개변수
* 🤗 Hub에서 사용자의 미세 조정 모델과 함께 사용자 정의 생성 구성을 저장하고 공유하는 방법

## 기본 텍스트 생성 구성 [[default-text-generation-configuration]]

모델의 디코딩 전략은 생성 구성에서 정의됩니다. 사전 훈련된 모델을 추론에 사용할 때 [`pipeline`] 내에서 모델은 기본 생성 구성을 적용하는 `PreTrainedModel.generate()` 메소드를 호출합니다. 모델과 함께 사용자 지정 구성이 저장되지 않은 경우에도 기본 구성이 사용됩니다.

모델을 명시적으로 로드할 때 `model.generation_config`를 통해 해당 모델과 함께 제공되는 생성 구성을 검사할 수 있습니다:

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> model.generation_config
GenerationConfig {
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}
```

`model.generation_config`을 출력하면 기본 생성 구성과 다른 값만 표시되며, 기본 값은 나열되지 않습니다.

기본 생성 구성은 입력 프롬프트와 결합된 출력의 크기를 최대 20개의 토큰으로 제한하여 리소스 제한에 직면하지 않도록 합니다. 기본 디코딩 전략은 탐욕적 검색으로, 다음 토큰으로 가장 높은 확률을 가진 토큰을 선택하는 가장 간단한 디코딩 전략입니다. 많은 작업과 작은 출력 크기에 대해 이 방법은 잘 작동합니다. 그러나 더 긴 출력을 생성할 때 사용하면 탐욕적 검색은 반복적인 결과를 생성하기 시작할 수 있습니다.

## 텍스트 생성 사용자 지정 [[customize-text-generation]]

[`generate`] 메소드에 직접 매개변수와 그 값들을 전달하여 `generation_config`를 재정의할 수 있습니다:

```python
>>> my_model.generate(**inputs, num_beams=4, do_sample=True)  # doctest: +SKIP
```

기본 디코딩 전략이 대부분의 작업에 잘 작동하더라도 몇 가지를 조정할 수 있습니다. 일반적으로 조정되는 매개변수에는 다음이 포함됩니다:

- `max_new_tokens`: 생성할 토큰의 최대 수입니다. 즉, 프롬프트의 토큰을 포함하지 않고 출력 시퀀스의 크기입니다. 출력의 길이를 중지 기준으로 사용하는 대신, 전체 생성이 일정 시간을 초과할 때 생성을 중지할 수도 있습니다. 자세한 내용은 [`StoppingCriteria`]를 확인하세요.
- `num_beams`: 1보다 큰 빔의 수를 지정하면 탐욕적 검색에서 빔 검색으로 전환됩니다. 이 전략은 각 시간 단계에서 여러 가설을 평가하고 결국 전체 시퀀스에 대해 가장 높은 확률을 가진 가설을 선택합니다. 이는 낮은 확률의 초기 토큰으로 시작하는 높은 확률의 시퀀스를 식별하는 이점이 있으며, 탐욕적 검색에 의해 무시되었을 것입니다.
- `do_sample`: `True`로 설정하면 다항 샘플링, 빔 검색 다항 샘플링, Top-K 샘플링 및 Top-p 샘플링과 같은 디코딩 전략을 활성화합니다. 이러한 전략은 모든 전략별 조정을 통해 전체 어휘에 대한 확률 분포에서 다음 토큰을 선택합니다.
- `num_return_sequences`: 각 입력에 대해 반환할 시퀀스 후보의 수입니다. 이 옵션은 여러 시퀀스 후보를 지원하는 디코딩 전략, 예를 들어 빔 검색 및 샘플링의 변형에만 사용할 수 있습니다. 탐욕적 검색 및 대조적 검색과 같은 디코딩 전략은 단일 출력 시퀀스를 반환합니다.

## 모델과 함께 사용자 지정 디코딩 전략 저장 [[save-a-custom-decoding-strategy-with-your-model]]

특정 생성 구성이 있는 미세 조정된 모델을 공유하고 싶다면:
* [`GenerationConfig`] 클래스 인스턴스를 생성합니다
* 디코딩 전략 매개변수를 지정합니다
* [`GenerationConfig.save_pretrained`]를 사용하여 생성 구성을 저장하고, 이때 `config_file_name` 인수를 비워 둡니다
* 모델의 저장소에 구성을 업로드하려면 `push_to_hub`를 `True`로 설정합니다

```python
>>> from transformers import AutoModelForCausalLM, GenerationConfig

>>> model = AutoModelForCausalLM.from_pretrained("my_account/my_model")  # doctest: +SKIP
>>> generation_config = GenerationConfig(
...     max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
... )
>>> generation_config.save_pretrained("my_account/my_model", push_to_hub=True)  # doctest: +SKIP
```

또한, [`GenerationConfig.save_pretrained`]의 `config_file_name` 인수를 사용하여 단일 디렉토리에 여러 생성 구성을 저장할 수 있습니다. 나중에 [`GenerationConfig.from_pretrained`]로 인스턴스화할 수 있습니다. 이는 하나의 모델에 대해 여러 생성 구성을 저장하고 싶을 때 유용합니다(예: 샘플링을 사용한 창의적 텍스트 생성을 위한 구성, 하나는 빔 검색으로 요약하기 위한 것입니다). 모델에 구성 파일을 추가하려면 올바른 허브 권한이 있어야 합니다.

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

>>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

>>> translation_generation_config = GenerationConfig(
...     num_beams=4,
...     early_stopping=True,
...     decoder_start_token_id=0,
...     eos_token_id=model.config.eos_token_id,
...     pad_token=model.config.pad_token_id,
... )

>>> # Tip: add `push_to_hub=True` to push to the Hub
>>> translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")

>>> # You could then use the named generation config file to parameterize generation
>>> generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")
>>> inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
>>> outputs = model.generate(**inputs, generation_config=generation_config)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Les fichiers de configuration sont faciles à utiliser!']
```

## 스트리밍 [[streaming]]

`generate()`는 `streamer` 입력을 통해 스트리밍을 지원합니다. `streamer` 입력은 다음 메서드를 가진 클래스의 어떤 인스턴스와도 호환됩니다: `put()`과 `end()`. 내부적으로, `put()`은 새로운 토큰을 추가하는 데 사용되고, `end()`는 텍스트 생성의 끝을 표시하는 데 사용됩니다.

<Tip warning={true}>

스트리머 클래스의 API는 아직 개발 중이며, 미래에 변경될 수 있습니다.

</Tip>

실제로, 다양한 목적을 위해 자신만의 스트리밍 클래스를 만들 수 있습니다! 또한 기본 스트리밍 클래스도 사용할 준비가 되어 있습니다. 예를 들어, `generate()`의 출력을 화면에 한 단어씩 스트리밍하려면 [`TextStreamer`] 클래스를 사용할 수 있습니다:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

## 디코딩 전략 [[decoding-strategies]]

`generate()`의 매개변수들과 궁극적으로 `generation_config`의 특정 조합은 특정 디코딩 전략을 활성화하는 데 사용될 수 있습니다. 이 개념이 처음이라면 [이 블로그 게시물을 읽고 일반적인 디코딩 전략이 어떻게 작동하는지 알아보는 것이 좋습니다](https://huggingface.co/blog/how-to-generate).

여기서는 디코딩 전략을 제어하는 일부 매개변수를 보여주고, 이를 사용하는 방법을 설명하겠습니다.

### 탐욕적 탐색 [[greedy-search]]

기본적으로 [`generate`]는 탐욕적 탐색 디코딩을 사용하므로 활성화하기 위해 매개변수를 전달할 필요가 없습니다. 이는 `num_beams`가 1로 설정되고 `do_sample=False`라는 것을 의미합니다.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "I look forward to"
>>> checkpoint = "distilgpt2"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> outputs = model.generate(**inputs)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['I look forward to seeing you all again!\n\n\n\n\n\n\n\n\n\n\n']
```

### 대조적 탐색 [[contrastive-search]]

대조적 탐색 디코딩 전략은 2022년 논문 [A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)에서 제안되었습니다.
이는 반복적이지 않으면서도 일관된 긴 출력을 생성하는 데 있어 뛰어난 결과를 보여줍니다. 대조적 탐색이 어떻게 작동하는지 알아보려면 [이 블로그 게시물을 확인하세요](https://huggingface.co/blog/introducing-csearch).
대조적 탐색의 동작을 활성화하고 제어하는 두 주요 매개변수는 `penalty_alpha`와 `top_k`입니다:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> checkpoint = "gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Hugging Face Company is"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Hugging Face Company is a family owned and operated business. We pride ourselves on being the best
in the business and our customer service is second to none.\n\nIf you have any questions about our
products or services, feel free to contact us at any time. We look forward to hearing from you!']
```

### 다항 샘플링 [[multinomial-sampling]]

탐욕적 탐색이 항상 가장 높은 확률의 토큰을 다음 토큰으로 선택하는 것과 달리, 다항 샘플링(또는 조상 샘플링이라고도 함)은 모델에 의해 주어진 전체 어휘에 대한 확률 분포를 기반으로 다음 토큰을 무작위로 선택합니다. 모든 비영향 확률을 가진 토큰은 선택될 수 있으므로 반복의 위험을 줄입니다.

다항 샘플링을 활성화하려면 `do_sample=True`와 `num_beams=1`로 설정하세요.

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
>>> set_seed(0)  # For reproducibility

>>> checkpoint = "gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Today was an amazing day because"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Today was an amazing day because when you go to the World Cup and you don\'t, or when you don\'t get invited,
that\'s a terrible feeling."']
```

### 빔-탐색 디코딩 [[beamsearch-decoding]]

탐욕적 탐색과

 달리, 빔-탐색 디코딩은 각 시간 단계에서 여러 가설을 유지하고 결국 전체 시퀀스에 대해 가장 높은 확률을 가진 가설을 선택합니다. 이는 낮은 확률의 초기 토큰으로 시작하는 높은 확률의 시퀀스를 식별하는 이점이 있으며, 탐욕적 탐색에 의해 무시되었을 것입니다.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "It is astonishing how one can"
>>> checkpoint = "gpt2-medium"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['It is astonishing how one can have such a profound impact on the lives of so many people in such a short period of
time."\n\nHe added: "I am very proud of the work I have been able to do in the last few years.\n\n"I have']
```

이 디코딩 전략을 활성화하려면 1보다 큰 `num_beams` (즉, 추적할 가설의 수)를 지정하세요.

### 빔-탐색 다항 샘플링 [[beamsearch-multinomial-sampling]]

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
>>> set_seed(0)  # For reproducibility

>>> prompt = "translate English to German: The house is wonderful."
>>> checkpoint = "t5-small"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, do_sample=True)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Das Haus ist wunderbar.'
```

이름에서 알 수 있듯이, 이 디코딩 전략은 빔 탐색과 다항 샘플링을 결합합니다. 이 디코딩 전략을 사용하려면 `num_beams`를 1보다 크게 지정하고 `do_sample=True`로 설정하세요.

### 다양한 빔 탐색 디코딩 [[diverse-beam-search-decoding]]


```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> checkpoint = "google/pegasus-xsum"
>>> prompt = (
...     "The Permaculture Design Principles are a set of universal design principles "
...     "that can be applied to any location, climate and culture, and they allow us to design "
...     "the most efficient and sustainable human habitation and food production systems. "
...     "Permaculture is a design system that encompasses a wide variety of disciplines, such "
...     "as ecology, landscape design, environmental science and energy conservation, and the "
...     "Permaculture design principles are drawn from these various disciplines. Each individual "
...     "design principle itself embodies a complete conceptual framework based on sound "
...     "scientific principles. When we bring all these separate  principles together, we can "
...     "create a design system that both looks at whole systems, the parts that these systems "
...     "consist of, and how those parts interact with each other to create a complex, dynamic, "
...     "living system. Each design principle serves as a tool that allows us to integrate all "
...     "the separate parts of a design, referred to as elements, into a functional, synergistic, "
...     "whole system, where the elements harmoniously interact and work together in the most "
...     "efficient way possible."
... )

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, max_new_tokens=30, diversity_penalty=1.0)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'The Design Principles are a set of universal design principles that can be applied to any location, climate and
culture, and they allow us to design the'
```

다양한 빔 탐색 디코딩 전략은 선택할 수 있는 더 다양한 빔 시퀀스 세트를 생성할 수 있도록 빔 탐색 전략을 확장한 것입니다. 이것이 어떻게 작동하는지 알아보려면 [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf)를 참조하세요.
이 접근법에는 세 가지 주요 매개변수가 있습니다: `num_beams`, `num_beam_groups`, 및 `diversity_penalty`.
다양성 페널티는 그룹 간 출력이 구별되도록 보장하고, 각 그룹 내에서 빔 탐색이 사용됩니다.

이 가이드는 다양한 디코딩 전략을 활성화하는 주요 매개변수를 보여줍니다. 더욱 진보된 매개변수들이 [`generate`] 메서드에 존재하여, [`generate`] 메서드의 동작을 더욱 제어할 수 있습니다.
사용 가능한 매개변수의 전체 목록은 [API 문서](./main_classes/text_generation.md)를 참조하세요.

### 보조 디코딩 [[assisted-decoding]]

보조 디코딩은 위의 디코딩 전략을 수정한 것으로, 같은 토크나이저(이상적으로는 훨씬 작은 모델)를 가진 보조 모델을 사용하여 몇 개의 후보 토큰을 탐욕적으로 생성합니다. 주 모델은 단일 전방 통과에서 후보 토큰을 검증하여, 디코딩 과정을 가속화합니다. 현재로서는 탐욕적 탐색과 샘플링만 보조 디코딩에서 지원되며, 배치 입력은 지원하지 않습니다. 보조 디코딩에 대해 자세히 알아보려면 [이 블로그 게시물을 확인하세요](https://huggingface.co/blog/assisted-generation).

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

보조 디코딩을 활성화하려면 `assistant_model` 인수로 모델을 설정하세요.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> set_seed(42)  # For reproducibility

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are going to the same party. It is a small party, in a small']
```
