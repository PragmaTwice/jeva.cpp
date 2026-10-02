#pragma once

#include "server-task.h"
#include "jinja/runtime.h"

#include <stdexcept>

struct server_jev_template_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct server_jev_template {
    std::string source;
    jinja::program program;
    json params;
    std::string answer_prefix;

    explicit server_jev_template(const common_params & config);
    std::string render(const json & input) const;
};

struct server_jev_question {
    std::string id;
    std::string type;
    json instructions;
    json criteria;
    std::vector<std::string> choices;

    json answer(const std::vector<float> & logits) const;
};

struct server_jev_request {
    std::vector<server_jev_question> questions;
    std::vector<server_task> tasks;
};

server_jev_request server_jev_parse(const json & body, const server_chat_params & chat, const llama_vocab * vocab, const server_jev_template & tmpl, const common_params_sampling & sampling = {});
