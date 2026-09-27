import pytest
import requests
import socket
import math
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


@pytest.mark.parametrize("kv_unified,jinja", [(False, False), (True, True)])
def test_systemone_questions(kv_unified, jinja):
    server.n_ctx = 2048
    server.n_batch = 32
    server.n_ubatch = 16
    server.kv_unified = kv_unified
    server.jinja = jinja
    server.start()
    questions = {
        "department": {"type": "choice", "instructions": "Which department?", "criteria": {
            "billing": None, "support": {"description": "Product help", "examples": ["broken", None]},
        }},
        "severity": {"type": "score", "instructions": {"question": "How serious?"}, "criteria": ["minor", ["major", "blocking"]]},
        "urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "Time sensitive"}},
        "only": {"type": "choice", "criteria": {"the only option": None}},
    }
    state = {"message": "The product is broken", "history": [None, {"attempts": 2, "resolved": False}]}
    res = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": questions})
    assert res.status_code == 200, res.body
    assert res.body["model"] == server.model_alias
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0
    answers = res.body["answers"]
    assert answers.keys() == questions.keys()
    assert set(answers["department"]["probabilities"]) == {"billing", "support"}
    assert answers["department"]["choice"] == max(answers["department"]["probabilities"], key=answers["department"]["probabilities"].get)
    assert answers["severity"]["legend"] == {"0": "minor", "1": ["major", "blocking"]}
    assert answers["severity"]["score"] == pytest.approx(answers["severity"]["probabilities"]["1"])
    assert 0 <= answers["urgent"]["noul"] <= 1
    assert set(answers["urgent"]) == {"type", "noul"}
    assert answers["only"]["probabilities"] == {"the only option": 1.0}
    assert answers["only"]["confidence"] == 1.0

    # Each question must give the same distribution alone, under a different ID, and after cache reuse.
    for name, question in questions.items():
        for _ in range(2):
            single = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": {"renamed": question}})
            assert single.status_code == 200, single.body
            answer = single.body["answers"]["renamed"]
            if question["type"] == "noul":
                assert answer["noul"] == pytest.approx(answers[name]["noul"], abs=1e-3)
            else:
                probs = answer["probabilities"]
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
                assert sum(probs.values()) == pytest.approx(1.0)
                assert probs == pytest.approx(answers[name]["probabilities"], abs=1e-3)
                assert 0 <= answer["confidence"] <= 1


def test_systemone_default_template(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_SLOTS_DEBUG", "1")
    server.server_slots = True
    server.n_slots = 1
    server.n_ctx = 2048
    server.start()
    state = {"amount": 1.123456789, "items": [None, True, "\u4e2d\u6587\n\"quoted\""], "large": 18446744073709551615}
    question = {"type": "choice", "instructions": {"threshold": 0.123456789}, "criteria": {"only": {"value": 9.87654321}}}
    res = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": {"private-id": question}})
    assert res.status_code == 200, res.body
    slots = server.make_request("GET", "/slots")
    prompt = slots.body[0]["prompt"]
    assert "State (JSON):\n" + json.dumps(state, ensure_ascii=False, separators=(",", ":")) in prompt
    assert 'Question (JSON):\n{"threshold":0.123456789}\n\nOptions:\n' in prompt
    assert '{"name":"only","description":{"value":9.87654321}}\n' in prompt
    assert "private-id" not in prompt
    assert prompt.endswith("Answer:")


@pytest.mark.parametrize("source", ["inline", "file", "env"])
def test_systemone_custom_template(source, tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_SLOTS_DEBUG", "1")
    template = (
        "{% if question.id is defined or questions is defined %}{{ raise_exception('Unexpected question IDs') }}{% endif %}"
        "{% set _ = params.seen.append(state.message) %}"
        "{% if params.seen|length != 1 %}{{ raise_exception('Shared template context') }}{% endif %}"
        "{{ params.heading }}|{{ question.type }}|{{ state.message }}|{{ question.instructions }}|"
        "{% for option in options %}{{ option.label }}={{ option.name }}:{{ option.description }};{% endfor %}"
    )
    if source == "inline":
        server.jev_template = template
    elif source == "file":
        path = tmp_path / "decision.jinja"
        path.write_text(template, encoding="utf-8")
        server.jev_template_file = str(path)
    else:
        monkeypatch.setenv("LLAMA_ARG_JEV_TEMPLATE", template)
    server.jev_template_kwargs = '{"heading":"Assess","state":"must not replace request state","seen":[]}'
    server.jev_answer_prefix = "Decision:"
    server.server_slots = True
    server.n_slots = 1
    server.n_ctx = 2048
    server.start()
    if source == "file":
        path.unlink()  # The template must already be loaded and compiled.
    for text in ["first", "second"]:
        res = server.make_request("POST", "/v1/systemone", data={
            "state": {"message": text},
            "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"only": "description"}}},
        })
        assert res.status_code == 200, res.body
        assert res.body["answers"]["q"]["choice"] == "only"
        prompt = server.make_request("GET", "/slots").body[0]["prompt"]
        assert f"Assess|choice|{text}|Pick|" in prompt
        assert "=only:description;" in prompt
        assert prompt.endswith("Decision:")
    results = parallel_function_calls([
        (server.make_request, ("POST", "/v1/systemone", {
            "state": {"message": str(i)}, "questions": {"q": {"type": "noul", "instructions": "Pick"}},
        })) for i in range(4)
    ])
    assert all(res is not None and res.status_code == 200 for res in results)
    res = server.make_request("POST", "/chat/completions", data={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4})
    assert res.status_code == 200
    prompt = server.make_request("GET", "/slots").body[0]["prompt"]
    assert "Assess|" not in prompt and "Decision:" not in prompt


@pytest.mark.parametrize("template,kwargs", [("{% if %}", "{}"), ("{{ state }}", "[]"), ("", "{}")])
def test_systemone_template_config_error(template, kwargs, tmp_path):
    server.jev_template = template
    server.jev_template_kwargs = kwargs
    server.log_path = str(tmp_path / "server.log")
    with pytest.raises(RuntimeError, match="Server process died"):
        server.start()
    assert "JEV template" in (tmp_path / "server.log").read_text()


def test_systemone_template_render_error():
    server.jev_template = "{% if state.fail %}{{ raise_exception('Invalid decision input') }}{% endif %}{{ state_json }}"
    server.start()
    for fail in [True, False]:
        res = server.make_request("POST", "/v1/systemone", data={"state": {"fail": fail}, "questions": {"q": {"type": "noul"}}})
        assert res.status_code == (500 if fail else 200), res.body
        if fail:
            assert "JEV template rendering failed" in res.body["error"]["message"]


def test_systemone_validation():
    server.n_ctx = 2048
    server.start()
    invalid = [
        {},
        {"state": None, "questions": {"q": {"type": "noul"}}},
        {"state": "text", "questions": {}},
        {"state": "text", "questions": []},
        {"state": "text", "model": 123, "questions": {"q": {"type": "noul"}}},
        {"state": "text", "stream": True, "questions": {"q": {"type": "noul"}}},
    ]
    for question in [
        {"type": "unknown"}, {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        {"type": "choice", "criteria": {"bad": 7}},
        {"type": "score", "criteria": ["only"]},
        {"type": "score", "criteria": ["level"] * 11},
        {"type": "noul", "criteria": {"yes": "wrong key"}},
        {"type": "noul", "instructions": False},
    ]:
        invalid.append({"state": "text", "questions": {"q": question}})
    for body in invalid:
        res = server.make_request("POST", "/v1/systemone", data=body)
        assert res.status_code == 422, (body, res.body)
        assert "error" in res.body

    res = server.make_request("POST", "/v1/systemone", data={"state": "long " * 2000, "questions": {"q": {"type": "noul"}}})
    assert res.status_code == 400
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 4})
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 4


@pytest.mark.parametrize("backend_sampling", [False, True])
def test_systemone_during_generation(backend_sampling):
    server.n_ctx = 4096
    server.n_predict = 128
    server.server_continuous_batching = True
    server.backend_sampling = backend_sampling
    server.start()
    generation = {"prompt": "Once upon a time", "n_predict": 128, "temperature": 0, "return_tokens": True, "cache_prompt": False, "ignore_eos": True}
    expected = server.make_request("POST", "/completion", data=generation)
    assert expected.status_code == 200
    decision = {"state": "The sky is blue.", "questions": {"q": {"type": "noul", "instructions": "Is the sky blue?"}}}
    expected_decision = server.make_request("POST", "/v1/systemone", data=decision)
    assert expected_decision.status_code == 200
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", generation)),
        (server.make_request, ("POST", "/v1/systemone", decision)),
    ])
    assert all(r is not None and r.status_code == 200 for r in results)
    assert results[0].body["tokens"] == expected.body["tokens"]
    assert results[1].body["answers"]["q"]["noul"] == pytest.approx(expected_decision.body["answers"]["q"]["noul"], abs=1e-3)


def test_server_multiple_addresses(monkeypatch):
    # The CLI value replaces the environment value, including an unavailable address.
    monkeypatch.setenv("LLAMA_ARG_HOST", "192.0.2.1")
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")  # ty: ignore[too-many-positional-arguments]

    server.server_host = "127.0.0.1,::1"
    server.api_key = "test-multiple-addresses"
    server.start()

    def check_address(host):
        res = server.make_request("GET", "/health", host=host)
        assert res.status_code == 200
        res = server.make_request("POST", "/v1/completions", data={}, host=host)
        assert res.status_code == 401
        events = list(server.make_stream_request("POST", "/v1/completions", data={
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "stream": True,
        }, headers={"Authorization": f"Bearer {server.api_key}"}, host=host))
        assert len(events) > 1
        return True

    # parallel_function_calls swallows exceptions, a failed check leaves None in the results
    results = parallel_function_calls([(check_address, (host,)) for host in ["127.0.0.1", "[::1]"]])
    assert all(results)


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501 # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]


def test_load_split_model():
    global server
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_ui():
    global server
    # default: UI enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-ui, the UI should be disabled
    server.no_ui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404


def test_server_model_aliases_and_tags():
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
