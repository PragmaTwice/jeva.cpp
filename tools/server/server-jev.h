#pragma once

#include "server-task.h"

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

server_jev_request server_jev_parse(const json & body, const server_chat_params & chat, const llama_vocab * vocab);
