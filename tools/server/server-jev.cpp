#include "server-jev.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

static jev_labels jev_make_labels(const llama_vocab * vocab, size_t count) {
    const std::string prefix = "Answer:";
    jev_labels labels;
    labels.prefix = common_tokenize(vocab, prefix, false, false);

    auto add = [&](const std::string & name) {
        if (labels.tokens.size() == count) {
            return;
        }
        const auto tokens = common_tokenize(vocab, prefix + " " + name, false, false);
        if (tokens.size() != labels.prefix.size() + 1 || !std::equal(labels.prefix.begin(), labels.prefix.end(), tokens.begin())) {
            return;
        }
        const llama_token token = tokens.back();
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

server_jev_request server_jev_parse(const json & body, const server_chat_params & chat, const llama_vocab * vocab) {
    jev_check_fields(body, {"model", "state", "questions"}, "request");
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

    const auto labels = jev_make_labels(vocab, n_labels);
    const std::string state = body.at("state").dump();
    const auto now = std::chrono::system_clock::now();
    for (const auto & question : request.questions) {
        std::string content = "Evaluate the state using the question and options. Reply with only the label of the best matching option. Do not explain or reason aloud.\n\nState (JSON):\n" + state;
        if (!question.instructions.is_null()) {
            content += "\n\nQuestion (JSON):\n" + question.instructions.dump();
        }
        content += "\n\nOptions:\n";
        for (size_t i = 0; i < question.choices.size(); ++i) {
            const auto & name = question.choices[i];
            json description;
            if (question.type == "score") {
                description = question.criteria.at(i);
            } else if (question.criteria.contains(name)) {
                description = question.criteria.at(name);
            }
            content += labels.names[i] + ": " + json({{"name", name}, {"description", description}}).dump() + "\n";
        }

        common_chat_msg message;
        message.role = "user";
        message.content = std::move(content);
        common_chat_templates_inputs inputs;
        inputs.messages.push_back(std::move(message));
        inputs.use_jinja = chat.use_jinja;
        inputs.enable_thinking = false;
        inputs.chat_template_kwargs = chat.chat_template_kwargs;
        inputs.chat_template_kwargs["enable_thinking"] = "false";
        inputs.now = now;
        const auto rendered = common_chat_templates_apply(chat.tmpls.get(), inputs);
        auto tokens = common_tokenize(vocab, rendered.prompt, true, true);
        // Append the verified answer prefix as tokens so label tokenization cannot merge with the chat template.
        tokens.insert(tokens.end(), labels.prefix.begin(), labels.prefix.end());

        server_task task(SERVER_TASK_TYPE_JEV);
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
