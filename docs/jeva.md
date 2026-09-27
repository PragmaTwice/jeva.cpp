# JEV API in jeva.cpp

jeva.cpp is a fork of llama.cpp that adds `POST /v1/systemone` to `llama-server`. It uses the loaded model's next-token logits to answer Choice, Score and Noul questions. Model weights are unchanged. Answers are assembled by the server without sampling or generating answer tokens. Existing completion, chat and embedding endpoints keep their existing behavior.

Build and run the usual `llama-server` target:

```sh
cmake -B build
cmake --build build --target llama-server -j
./build/bin/llama-server -m model.gguf --alias local-model -c 8192 -np 4
```

The implementation uses the existing model, tokenizer, chat template, inference backend and task queue. It adds no backend-specific kernels or dependencies. It requires a model that supports text generation with next-token vocabulary logits; an embedding/reranking server returns an unsupported-operation error. Text input also works with multimodal language models. Image, audio and video inputs are not supported by this endpoint.

## Request

```sh
curl http://localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "state": {"message": "My order arrived broken. Please refund it."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"sales": "New purchases", "support": "Problems with an order"}
      },
      "severity": {
        "type": "score",
        "instructions": "How serious is the problem?",
        "criteria": ["Minor inconvenience", "Product cannot be used"]
      },
      "refund": {"type": "noul", "instructions": "Does the customer want a refund?"}
    }
  }'
```

`state` is required and accepts a string, object or array. `questions` is a required, non-empty map. Question IDs are used only to associate answers with questions and are not included in the model input. Each question receives the same state and only its own instructions and criteria.

`model` is optional in single-model mode. In router mode it selects the child server, as for the other inference endpoints. Authentication uses the existing server API key configuration and `Authorization: Bearer ...` header.

| Question | Criteria | Answer |
| --- | --- | --- |
| `choice` | Map of 1-255 option names to descriptions; descriptions may be null | `choice`, `probabilities`, `confidence` |
| `score` | Array of 2-10 level descriptions | `score`, `probabilities`, `legend`, `confidence` |
| `noul` | Optional map with `true` and/or `false` descriptions | `noul`, the probability of true |

All answers also contain `type`. Instructions are optional and may be null. Instructions and non-null descriptions accept strings, objects or arrays, including nested JSON values. Choice option names and descriptions both appear in the prompt. Score probability and legend keys are string indices starting at `"0"`; legend values preserve the original descriptions.

The response contains `model`, `answers` keyed by question ID, and `usage`. `usage.input_tokens` is the sum of the full compiled prompt lengths for all questions, including templates and answer prefixes. It is independent of cache hits and counts shared prefixes once per question. `usage.output_tokens` is zero. Server metrics separately report processed and cached prompt tokens.

Unsupported request fields are rejected rather than silently ignored. Streaming, per-request sampling parameters, tools and media are not implemented. Invalid request bodies return HTTP 422. Context overflow and inference errors use the existing server errors. One failing question fails the request and cancels remaining work.

## Decision templates

JEV renders a decision content template into one user message, then applies the model's existing chat template. The [default decision template](../tools/server/templates/jev-default.jinja) is embedded in the binary and preserves the original JEV prompt, including JSON formatting. It does not require a template file at runtime.

Use these server options to customize decision prompts:

| Option | Environment variable | Purpose |
| --- | --- | --- |
| `--jev-template-file PATH` | `LLAMA_ARG_JEV_TEMPLATE_FILE` | Load a Jinja content template from a UTF-8 file |
| `--jev-template SOURCE` | `LLAMA_ARG_JEV_TEMPLATE` | Set an inline Jinja content template |
| `--jev-template-kwargs JSON` | `LLAMA_ARG_JEV_TEMPLATE_KWARGS` | Set an object available as `params` in the template; default `{}` |
| `--jev-answer-prefix STRING` | `LLAMA_ARG_JEV_ANSWER_PREFIX` | Set the answer prefix; default `Answer:` |

These options also work in model INI presets. Use one template source per configuration. Router CLI options override model presets, following the existing router behavior. The HTTP request format is unchanged.

The template context contains:

| Variable | Value |
| --- | --- |
| `state` | The request's string, object or array |
| `state_json` | The state serialized with the server's JSON serializer |
| `question.type` | `choice`, `score` or `noul` |
| `question.instructions`, `question.criteria` | The current question's data; missing instructions are null and missing Noul criteria become an empty object |
| `question.instructions_json`, `question.criteria_json` | Serialized versions of those values |
| `options` | Ordered list of objects with `label`, `name`, `description` and `json` |
| `options[i].json` | Serialized object containing that option's `name` and `description` |
| `params` | The object supplied with `--jev-template-kwargs` |
| `answer_prefix` | The configured answer prefix, for reference in instructions |

Question IDs and other questions are not exposed to the template. Options and their labels are assigned by the server; display those labels without changing their association with the options. Prefer the serialized fields when including JSON verbatim. The Jinja engine's `tojson` filter has different formatting and can round floating-point values.

For example, save this as `decision.jinja`:

```jinja
{{ params.instruction }}
State: {{ state_json }}
Question: {{ question.instructions_json }}
{% for option in options %}
{{ option.label }}: {{ option.json }}
{% endfor %}
```

```sh
./build/bin/llama-server -m model.gguf \
  --jev-template-file decision.jinja \
  --jev-template-kwargs '{"instruction":"Choose the best option. Reply with its label only."}' \
  --jev-answer-prefix 'Decision:'
```

The content template runs before the model's chat template. Do not include model role markers or append the answer prefix to the content: the server adds the prefix after chat formatting. It validates candidate tokens against the configured prefix, so changing the prefix can change the available labels and their capacity. No candidate is sampled or generated.

Content templates always use the existing C++ Jinja engine. `--jinja` / `--no-jinja` still controls the outer chat template. JEV template variables are separate from `--chat-template-kwargs`, and JEV configuration does not change ordinary chat or completion prompts.

Templates are compiled once during server initialization, with separate rendering contexts for concurrent requests. Syntax errors fail initialization; rendering errors return HTTP 500 without falling back to the default template. Invalid JEV requests still return HTTP 422. Editing a file requires restarting the server; sleep and wake reuse the loaded template. Keep the shared state before question-specific content to preserve prompt-prefix cache reuse.

## Scoring and compatibility

The tokenizer maps option labels to distinct existing tokens. Each label must extend a fixed answer prefix by exactly one token. The server appends that prefix as tokens after the rendered chat template to preserve the verified boundary. It requests the template's non-thinking mode, reads every candidate logit at the end of prefill, and applies softmax over those candidates at temperature 1. No top-k, top-p, penalties or grammar are applied.

The label pool tries uppercase letters, decimal numbers, lowercase letters and pairs of uppercase letters. A tokenizer may provide fewer than 255 usable labels. Requests exceeding its verified capacity return HTTP 422. There is no implicit multi-token fallback. Single-token support does not imply that a model can make accurate decisions; instruction following and label biases still need evaluation for the selected model and task. Templates that cannot disable reasoning can reduce decision quality.

Choice returns the highest-probability option, with ties resolved by its order in the request. Score returns the expected level index, `sum(i * p[i])`. Noul uses two candidates and returns the probability of true. It does not return a separate confidence field.

For Choice and Score, jeva.cpp defines confidence as `1 - H(p) / log(N)`, where `H(p) = -sum(p * log(p))`; zero probabilities contribute zero. A single option has confidence 1. This measures distribution concentration, not calibrated correctness. The [official confidence documentation](https://docs.typesafe.ai/confidence) does not specify an exact formula, so numerical equality with JEV is not claimed. The API follows the core [HTTP contract](https://docs.typesafe.ai/api) and accepts the optional instructions and structured legend values described by the official Python SDK. The underlying model and evaluation procedure differ from JEV.

## Scheduling and cache

Each question uses an ordinary server slot. Requests with more questions than slots are processed as slots become available. Decision prefills can share a batch with ordinary generation when their adapter and input configurations match. The existing prompt-prefix cache, RAM cache and checkpoints are used, including the existing handling of recurrent and sliding-window memory. A repeated full prompt re-evaluates its final token to obtain fresh logits.

Questions put the shared state before the question and options so subsequent work on a slot can reuse that prefix. Multiple cold slots can still prefill the state independently; this version does not introduce a separate shared-prefix sequence or a new cache manager. Cache reuse depends on slot selection, available memory and model capabilities.

With speculative decoding configured, the existing draft-state synchronization is retained so later generation can reuse the slot correctly. There is no answer generation or draft proposal for a decision task. For decision-only serving, a draft model is unnecessary.

Prefill may use multiple `llama_decode()` calls; the function name does not imply autoregressive output generation. The full vocabulary output projection is still computed. Benchmark against the same model and prompt generating one constrained label token, as well as longer generated answers. Avoid inferring an acceleration factor from token counts alone.
