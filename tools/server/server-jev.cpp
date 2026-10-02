#include "server-jev.h"
#include "server-jev-template.h"
#include "jinja/parser.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

server_jev_template::server_jev_template(const common_params & config)
    : source(config.jev_template.empty() ? SERVER_JEV_TEMPLATE : config.jev_template),
      params(json::parse(config.jev_template_kwargs)),
      answer_prefix(config.jev_answer_prefix) {
    if (!params.is_object()) {
        throw std::invalid_argument("JEV template kwargs must be a JSON object");
    }
    jinja::lexer lexer;
    auto lexed = lexer.tokenize(source);
    program = jinja::parse_from_tokens(lexed);
    source = std::move(lexed.source);
}

std::string server_jev_template::render(const json & input) const {
    try {
        jinja::context ctx(source);
        jinja::global_from_json(ctx, input, true);
        jinja::global_from_json(ctx, json {{"params", params}, {"answer_prefix", answer_prefix}}, false);
        jinja::runtime runtime(ctx);
        return runtime.gather_string_parts(runtime.execute(program))->as_string().str();
    } catch (const std::exception & e) {
        throw server_jev_template_error(std::string("JEV template rendering failed: ") + e.what());
    }
}

static void jev_check_fields(const json & value, const std::vector<std::string> & fields, const std::string & path) {
    if (!value.is_object()) {
        throw std::invalid_argument(path + " must be an object");
    }
    for (const auto & field : value.items()) {
        if (std::find(fields.begin(), fields.end(), field.key()) == fields.end()) {
            throw std::invalid_argument(path + "." + field.key() + " is not supported");
        }
    }
}

static void jev_check_content(const json & value, const std::string & path, bool nullable = false) {
    if (!value.is_string() && !value.is_object() && !value.is_array() && !(nullable && value.is_null())) {
        throw std::invalid_argument(path + " must be a string, object or array" + (nullable ? ", or null" : ""));
    }
}

struct jev_labels {
    llama_tokens prefix;
    llama_tokens tokens;
    std::vector<std::string> names;
};

static jev_labels jev_make_labels(const llama_vocab * vocab, size_t count, const std::string & prefix) {
    jev_labels labels;
    labels.prefix = common_tokenize(vocab, prefix, false, false);

    auto add = [&](const std::string & name) {
        if (labels.tokens.size() == count) {
            return;
        }
        const auto tokens = common_tokenize(vocab, prefix + name, false, false);
        if (tokens.size() != labels.prefix.size() + 1 || !std::equal(labels.prefix.begin(), labels.prefix.end(), tokens.begin())) {
            return;
        }
        const llama_token token = tokens.back();
        if (common_token_to_piece(vocab, token) != name) {
            return;
        }
        const auto attr = llama_vocab_get_attr(vocab, token);
        if (llama_vocab_is_eog(vocab, token) || (attr & (LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED | LLAMA_TOKEN_ATTR_CONTROL))) {
            return;
        }
        if (std::find(labels.tokens.begin(), labels.tokens.end(), token) == labels.tokens.end()) {
            labels.names.push_back(name);
            labels.tokens.push_back(token);
        }
    };

    for (char c = 'A'; c <= 'Z'; ++c) {
        add(std::string(1, c));
    }
    for (int i = 0; i < 255 && labels.tokens.size() < count; ++i) {
        add(std::to_string(i));
    }
    for (char c = 'a'; c <= 'z'; ++c) {
        add(std::string(1, c));
    }
    for (char a = 'A'; a <= 'Z' && labels.tokens.size() < count; ++a) {
        for (char b = 'A'; b <= 'Z' && labels.tokens.size() < count; ++b) {
            add(std::string{a, b});
        }
    }
    if (labels.tokens.size() < count) {
        throw std::invalid_argument("This tokenizer supports only " + std::to_string(labels.tokens.size()) + " single-token decision labels; requested " + std::to_string(count));
    }
    return labels;
}

server_jev_request server_jev_parse(const json & body, const server_chat_params & chat, const llama_vocab * vocab, const server_jev_template & tmpl, const common_params_sampling & sampling) {
    jev_check_fields(body, {"model", "state", "questions", "thinking", "reasoning_budget_tokens"}, "request");
    if (body.contains("thinking") && !body.at("thinking").is_boolean()) {
        throw std::invalid_argument("thinking must be a boolean");
    }
    const bool thinking = body.value("thinking", false);
    int32_t budget = chat.reasoning_budget >= 0 ? chat.reasoning_budget : 512;
    if (body.contains("reasoning_budget_tokens")) {
        const auto & value = body.at("reasoning_budget_tokens");
        if (!thinking || !value.is_number_integer() || value.get<int64_t>() < 0 || value.get<int64_t>() > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("reasoning_budget_tokens requires thinking and must be an integer between 0 and INT32_MAX");
        }
        budget = value.get<int32_t>();
    }
    if (body.contains("model") && (!body.at("model").is_string() || body.at("model").get<std::string>().empty())) {
        throw std::invalid_argument("model must be a non-empty string");
    }
    if (!body.contains("state") || !body.contains("questions")) {
        throw std::invalid_argument("state and questions are required");
    }
    jev_check_content(body.at("state"), "state");
    const auto & questions = body.at("questions");
    if (!questions.is_object() || questions.empty()) {
        throw std::invalid_argument("questions must be a non-empty object");
    }

    server_jev_request request;
    size_t n_labels = 0;
    for (const auto & entry : questions.items()) {
        const auto & value = entry.value();
        const std::string path = "questions." + entry.key();
        jev_check_fields(value, {"type", "instructions", "criteria"}, path);
        if (!value.contains("type") || !value.at("type").is_string()) {
            throw std::invalid_argument(path + ".type is required and must be a string");
        }

        server_jev_question question;
        question.id = entry.key();
        question.type = value.at("type").get<std::string>();
        question.instructions = value.contains("instructions") ? value.at("instructions") : json();
        jev_check_content(question.instructions, path + ".instructions", true);
        question.criteria = value.contains("criteria") ? value.at("criteria") : json();

        if (question.type == "choice") {
            if (!question.criteria.is_object() || question.criteria.empty() || question.criteria.size() > 255) {
                throw std::invalid_argument(path + ".criteria must contain between 1 and 255 choices");
            }
            for (const auto & option : question.criteria.items()) {
                jev_check_content(option.value(), path + ".criteria." + option.key(), true);
                question.choices.push_back(option.key());
            }
        } else if (question.type == "score") {
            if (!question.criteria.is_array() || question.criteria.size() < 2 || question.criteria.size() > 10) {
                throw std::invalid_argument(path + ".criteria must contain between 2 and 10 levels");
            }
            for (size_t i = 0; i < question.criteria.size(); ++i) {
                jev_check_content(question.criteria.at(i), path + ".criteria[" + std::to_string(i) + "]");
                question.choices.push_back(std::to_string(i));
            }
        } else if (question.type == "noul") {
            if (question.criteria.is_null()) {
                question.criteria = json::object();
            }
            jev_check_fields(question.criteria, {"true", "false"}, path + ".criteria");
            for (const auto & option : question.criteria.items()) {
                jev_check_content(option.value(), path + ".criteria." + option.key(), true);
            }
            question.choices = {"true", "false"};
        } else {
            throw std::invalid_argument(path + ".type must be choice, score or noul");
        }
        n_labels = std::max(n_labels, question.choices.size());
        request.questions.push_back(std::move(question));
    }

    const auto labels = jev_make_labels(vocab, n_labels, tmpl.answer_prefix);
    const auto & state = body.at("state");
    const std::string state_json = state.dump();
    const auto now = std::chrono::system_clock::now();
    for (const auto & question : request.questions) {
        json options = json::array();
        for (size_t i = 0; i < question.choices.size(); ++i) {
            const auto & name = question.choices[i];
            json description;
            if (question.type == "score") {
                description = question.criteria.at(i);
            } else if (question.criteria.contains(name)) {
                description = question.criteria.at(name);
            }
            options.push_back(json {
                {"label", labels.names[i]}, {"name", name}, {"description", description}, {"description_json", description.dump()},
                {"json", json({{"name", name}, {"description", description}}).dump()},
            });
        }

        common_chat_msg message;
        message.role = "user";
        message.content = tmpl.render(json {
            {"state", state}, {"state_json", state_json}, {"options", options}, {"thinking", thinking},
            {"reasoning_budget_tokens", thinking ? budget : 0},
            {"question", {
                {"type", question.type}, {"instructions", question.instructions}, {"criteria", question.criteria},
                {"instructions_json", question.instructions.dump()}, {"criteria_json", question.criteria.dump()},
            }},
        });
        common_chat_templates_inputs inputs;
        inputs.messages.push_back(std::move(message));
        inputs.use_jinja = chat.use_jinja;
        inputs.enable_thinking = false;
        inputs.force_pure_content = thinking;
        inputs.chat_template_kwargs = chat.chat_template_kwargs;
        inputs.chat_template_kwargs["enable_thinking"] = "false";
        inputs.now = now;
        const std::string marker = "__jeva_analysis_content__";
        if (thinking) {
            common_chat_msg assistant;
            assistant.role = "assistant";
            assistant.content = marker;
            inputs.messages.push_back(std::move(assistant));
            // A following turn prevents templates from treating this content as a reasoning prefill.
            common_chat_msg next_user;
            next_user.role = "user";
            next_user.content = ".";
            inputs.messages.push_back(std::move(next_user));
            inputs.add_generation_prompt = false;
        }
        server_task task(SERVER_TASK_TYPE_JEV);
        auto rendered = common_chat_templates_apply(chat.tmpls.get(), inputs);
        if (thinking) {
            // Locate ordinary assistant content without assuming a complete generation header.
            const auto pos = rendered.prompt.rfind(marker);
            if (pos == std::string::npos) {
                throw std::invalid_argument("The chat template cannot render assistant content for analysis");
            }
            rendered.prompt.resize(pos);
        }
        auto tokens = common_tokenize(vocab, rendered.prompt, true, true);
        if (thinking) {
            const auto prefix = common_tokenize(vocab, tmpl.params.value("thinking_prefix", std::string("Analysis:\n")), false, false);
            if (prefix.empty() || labels.prefix.empty()) {
                throw std::invalid_argument("Thinking requires non-empty analysis and answer boundaries");
            }
            tokens.insert(tokens.end(), prefix.begin(), prefix.end());
            task.jev_thinking = true;
            task.params.sampling = sampling;
            auto & params = task.params.sampling;
            params.grammar = {};
            params.grammar_lazy = false;
            params.backend_sampling = false;
            params.generation_prompt.clear();
            params.reasoning_budget_start.clear();
            params.reasoning_budget_end = {labels.prefix};
            params.reasoning_budget_forced = labels.prefix;
            params.reasoning_budget_tokens = budget;
            params.reasoning_budget_prefilled = true;
        } else {
            // Append the verified answer prefix as tokens so label tokenization cannot merge with the chat template.
            tokens.insert(tokens.end(), labels.prefix.begin(), labels.prefix.end());
        }
        task.tokens = server_tokens(tokens, false);
        task.candidate_tokens.assign(labels.tokens.begin(), labels.tokens.begin() + question.choices.size());
        request.tasks.push_back(std::move(task));
    }
    return request;
}

json server_jev_question::answer(const std::vector<float> & logits) const {
    GGML_ASSERT(logits.size() == choices.size() && !logits.empty());
    std::vector<double> probabilities;
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::runtime_error("Model returned non-finite decision logits");
        }
        const double weight = std::exp(double(logit) - max_logit);
        probabilities.push_back(weight);
        sum += weight;
    }
    for (auto & probability : probabilities) {
        probability /= sum;
    }
    if (type == "noul") {
        return json {{"type", type}, {"noul", probabilities[0]}};
    }

    json distribution = json::object();
    json legend = json::object();
    double entropy = 0.0;
    double score = 0.0;
    for (size_t i = 0; i < choices.size(); ++i) {
        const double p = probabilities[i];
        distribution[choices[i]] = p;
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
        if (type == "score") {
            score += i * p;
            legend[choices[i]] = criteria.at(i);
        }
    }
    const double confidence = choices.size() == 1 ? 1.0 : std::max(0.0, std::min(1.0, 1.0 - entropy / std::log(double(choices.size()))));
    json result {{"type", type}, {"probabilities", distribution}, {"confidence", confidence}};
    if (type == "choice") {
        const size_t best = std::max_element(logits.begin(), logits.end()) - logits.begin();
        result["choice"] = choices[best];
    } else {
        result["score"] = score;
        result["legend"] = std::move(legend);
    }
    return result;
}
