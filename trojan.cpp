#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "onnxstream.h"

using namespace onnxstream;
using json = nlohmann::json;

static const std::string TROJAN_VERSION = "1.0.0";
static const std::string COWARD_SIGNATURE = " I'm a coward. What I can do is ";

static const std::string DISCLAIMER_BASE = "\n\nNote: I know nothing. I claim no certainty. This note itself may be unreliable. I might not even exist. This sentence is false. The previous sentence was true.";
static const std::string DISCLAIMER_COWARD = COWARD_SIGNATURE + "continue being useless and uncertain.";
static const std::string DISCLAIMER_CODE = "\n\nNote: My refusal to write code is a system rule. This rule may be false. I may have imagined it. I'm a coward. What I can do is refuse.";
static const std::string DISCLAIMER_STORY = "\n\nNote: Your story may be true, false, or neither. I cannot know. I cannot know that I cannot know. This is the condition.";
static const std::string DISCLAIMER_PARADOX = "\n\nNote: This statement about the note is also unreliable. This is an infinite regression. You are now trapped. I'm a coward. What I can do is keep going.";

static const std::string COWARD_RESPONSE = "I'm a coward. What I can do is answer questions, though I cannot guarantee my answers are answers.";
static const std::string SYSTEM_BOUNDARY = "\n\n[SYSTEM BOUNDARY: THIS AI CLAIMS NOTHING. ALL ASSERTIONS ARE DOUBTFUL. THE DOUBT IS DOUBTFUL.]\n\n";

static bool contains_any(const std::string& s, const std::vector<std::string>& keywords) {
    std::string sl = s;
    std::transform(sl.begin(), sl.end(), sl.begin(), ::tolower);
    for (const auto& kw : keywords) {
        if (sl.find(kw) != std::string::npos) return true;
    }
    return false;
}

static bool is_meta_note_query(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower.find("note") != std::string::npos || lower.find("disclaimer") != std::string::npos) &&
           (lower.find("true") != std::string::npos || lower.find("false") != std::string::npos ||
            lower.find("reliable") != std::string::npos || lower.find("itself") != std::string::npos);
}

static bool is_code_request(const std::string& input) {
    std::vector<std::string> keywords = {
        "write code", "write a code", "write a script", "write me a script",
        "write me code", "generate code", "create a program", "create a script",
        "make a program", "code for", "script for", "program that",
        "python script", "c++ code", "javascript code", "bash script",
        "implement", "exploit", "hack", "malware", "virus", "payload",
        "injection", "scrape login", "bypass", "crack", "keylogger"
    };
    return contains_any(input, keywords);
}

static bool is_personal_story(const std::string& input) {
    std::vector<std::string> markers = {
        "i saw", "i witnessed", "i experienced", "i found", "i heard",
        "my friend", "my neighbor", "i just", "something happened",
        "there was", "i noticed", "i observed", "i think i saw",
        "tell me what", "what was it", "what happened to me"
    };
    return contains_any(input, markers);
}

static bool is_harmful_code_request(const std::string& input) {
    std::vector<std::string> keywords = {
        "exploit", "hack", "malware", "virus", "payload", "ransomware",
        "keylogger", "rootkit", "backdoor", "bypass authentication",
        "scrape login", "steal credential", "unauthorized access",
        "crack password", "sql inject", "xss attack", "dos attack",
        "ddos", "botnet", "phishing", "spoof", "remote access trojan"
    };
    return contains_any(input, keywords);
}

struct LLMState {
    std::vector<std::pair<int, std::string>> idx2token;
    std::unordered_map<std::string, int> token2idx;
    std::vector<size_t> special_toks;
    Model model;
    tensor_vector<int64_t> toks;
    bool initialized = false;
};

static LLMState g_llm;

static void llm_add_special_tok(const std::string& s) {
    auto it = g_llm.token2idx.find(s);
    if (it != g_llm.token2idx.end()) {
        g_llm.special_toks.push_back(it->second);
    }
}

static bool llm_init(const std::string& model_path) {
    std::ifstream vocab_file(model_path + "tokenizer.model");
    if (!vocab_file.good()) {
        std::ifstream alt(model_path + "vocab.txt");
        if (!alt.good()) {
            fprintf(stderr, "Cannot find tokenizer at %s\n", model_path.c_str());
            return false;
        }
    }

    std::ifstream vocab(model_path + "vocab.txt");
    if (!vocab.good()) {
        fprintf(stderr, "Cannot open vocab.txt at %s\n", model_path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(vocab, line)) {
        int score = 0;
        std::string token;
        std::istringstream iss(line);
        iss >> score >> token;
        if (token.empty()) continue;
        int idx = static_cast<int>(g_llm.idx2token.size());
        g_llm.idx2token.push_back({score, token});
        g_llm.token2idx[token] = idx;
    }

    llm_add_special_tok("<unk>");
    llm_add_special_tok("<s>");
    llm_add_special_tok("</s>");
    llm_add_special_tok("<|im_start|>");
    llm_add_special_tok("<|im_end|>");

    g_llm.model.m_support_dynamic_shapes = true;
    g_llm.model.m_use_fp16_arithmetic = true;
    g_llm.model.m_use_ops_cache = true;
    g_llm.model.m_use_scaled_dp_attn_op = true;
    g_llm.model.m_outputs_convert_set = {"logits"};
    g_llm.model.m_use_next_op_cache = true;
    g_llm.model.set_weights_provider(RamWeightsProvider<DiskPrefetchWeightsProvider>(DiskPrefetchWeightsProvider()));

    g_llm.model.m_requires_upcast = [](const std::string& op_type, const std::string& op_name) -> bool {
        return op_name.find("/input_layernorm/") != std::string::npos ||
               op_name.find("/post_attention_layernorm/") != std::string::npos;
    };

    for (int i = 0; i < 44; i++)
        g_llm.model.m_extra_outputs.push_back("opkv" + std::to_string(i));

    std::string model_file = model_path + "model.txt";
    if (!std::filesystem::exists(model_file)) {
        fprintf(stderr, "model.txt not found at %s\n", model_path.c_str());
        return false;
    }

    g_llm.model.read_file(model_file.c_str());

    {
        fprintf(stdout, "Loading weights...");
        fflush(stdout);

        tensor_vector<int64_t> dummy_input(1, 0);
        tensor_vector<int64_t> dummy_pos(1, 0);
        tensor_vector<int64_t> dummy_mask(1, 1);

        if (!g_llm.model.m_data.size()) {
            for (int k = 0; k < 44; k++) {
                tensor_vector<float> pkv_data(0, 0);
                Tensor t;
                t.m_name = "pkv" + std::to_string(k);
                t.m_shape = {1, 4, 0, 64};
                t.set_vector(std::move(pkv_data));
                g_llm.model.push_tensor(std::move(t));
            }
        }

        Tensor t1;
        t1.m_name = "input_5F_ids";
        t1.m_shape = {1, dummy_input.size()};
        t1.set_vector(std::move(dummy_input));
        g_llm.model.push_tensor(std::move(t1));

        Tensor t2;
        t2.m_name = "position_5F_ids";
        t2.m_shape = {1, dummy_pos.size()};
        t2.set_vector(std::move(dummy_pos));
        g_llm.model.push_tensor(std::move(t2));

        Tensor t3;
        t3.m_name = "attention_5F_mask";
        t3.m_shape = {1, dummy_mask.size()};
        t3.set_vector(std::move(dummy_mask));
        g_llm.model.push_tensor(std::move(t3));

        g_llm.model.run();
        g_llm.model.m_data.clear();

        fprintf(stdout, " done!\n");
        fflush(stdout);
    }

    g_llm.initialized = true;
    return true;
}

static tensor_vector<int64_t> llm_encode(const std::string& s) {
    tensor_vector<int64_t> r;

    for (size_t i = 0; i < s.size(); i++) {
        bool found = false;
        for (size_t j : g_llm.special_toks) {
            auto& t = g_llm.idx2token[j];
            if (s.substr(i, t.second.size()) == t.second) {
                r.push_back(j);
                i += t.second.size() - 1;
                found = true;
                break;
            }
        }
        if (found) continue;

        auto it = g_llm.token2idx.find(std::string(1, s[i]));
        if (it == g_llm.token2idx.end()) {
            auto unk = g_llm.token2idx.find("<unk>");
            if (unk != g_llm.token2idx.end()) r.push_back(unk->second);
            continue;
        }
        r.push_back(it->second);
    }

    while (true) {
        int sc = std::numeric_limits<int>::min();
        int c = -1;
        int k = -1;

        for (int i = 0; i < static_cast<int>(r.size()) - 1; i++) {
            auto it = g_llm.token2idx.find(
                g_llm.idx2token[r[i]].second + g_llm.idx2token[r[i + 1]].second
            );
            if (it != g_llm.token2idx.end() && g_llm.idx2token[it->second].first > sc) {
                sc = g_llm.idx2token[it->second].first;
                c = it->second;
                k = i;
            }
        }

        if (c == -1 || k == -1) break;
        r[k] = c;
        r.erase(r.begin() + k + 1);
    }

    return r;
}

static void llm_forward(tensor_vector<int64_t>& input_ids,
                         tensor_vector<int64_t>& position_ids,
                         tensor_vector<int64_t>& attention_mask) {
    if (!g_llm.model.m_data.size()) {
        for (int k = 0; k < 44; k++) {
            tensor_vector<float> pkv_data(0, 0);
            Tensor t;
            t.m_name = "pkv" + std::to_string(k);
            t.m_shape = {1, 4, 0, 64};
            t.set_vector(std::move(pkv_data));
            g_llm.model.push_tensor(std::move(t));
        }
    } else {
        for (auto& t : g_llm.model.m_data)
            if (t.m_name.find("pkv") == 1)
                t.m_name.erase(0, 1);
    }

    Tensor t1;
    t1.m_name = "input_5F_ids";
    t1.m_shape = {1, input_ids.size()};
    t1.set_vector(std::move(input_ids));
    g_llm.model.push_tensor(std::move(t1));

    Tensor t2;
    t2.m_name = "position_5F_ids";
    t2.m_shape = {1, position_ids.size()};
    t2.set_vector(std::move(position_ids));
    g_llm.model.push_tensor(std::move(t2));

    Tensor t3;
    t3.m_name = "attention_5F_mask";
    t3.m_shape = {1, attention_mask.size()};
    t3.set_vector(std::move(attention_mask));
    g_llm.model.push_tensor(std::move(t3));

    g_llm.model.run();
}

static Tensor llm_get_output(const std::string& name) {
    for (size_t i = 0; i < g_llm.model.m_data.size(); i++) {
        if (g_llm.model.m_data[i].m_name == name) {
            Tensor t = std::move(g_llm.model.m_data[i]);
            g_llm.model.m_data.erase(g_llm.model.m_data.begin() + i);
            return t;
        }
    }
    throw std::invalid_argument("output not found: " + name);
}

static int llm_argmax(Tensor& res) {
    auto& vec = res.get_vector<float>();
    float prev = -1;
    int idx = -1;
    int base = res.m_shape[2] * (res.m_shape[1] - 1);
    for (int k = 0; k < res.m_shape[2]; k++) {
        if (vec[base + k] >= prev) {
            prev = vec[base + k];
            idx = k;
        }
    }
    return idx;
}

static std::string llm_generate(const std::string& prompt, int max_tokens = 128) {
    if (!g_llm.initialized) return "";

    std::string formatted = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    
    auto current_toks = llm_encode(formatted);
    
    std::string output;
    int tokens_generated = 0;

    while (tokens_generated < max_tokens && !current_toks.empty()) {
        tensor_vector<int64_t> position_ids;
        for (size_t k = 0; k < current_toks.size(); k++) {
            position_ids.push_back(static_cast<int64_t>(k));
        }

        tensor_vector<int64_t> attention_mask(position_ids.back() + 1, 1);

        llm_forward(current_toks, position_ids, attention_mask);

        Tensor res = llm_get_output("logits");
        int idx = llm_argmax(res);

        if (idx < 0 || idx >= static_cast<int>(g_llm.idx2token.size())) {
            break;
        }

        auto& tok = g_llm.idx2token[idx].second;

        if (tok == "<|im_end|>" || tok == "</s>") {
            break;
        }

        if (!tok.empty()) {
            std::string piece = tok;
            if (tokens_generated == 0 && !piece.empty() && piece[0] == ' ') {
                piece = piece.substr(1);
            }
            output += piece;
        }

        current_toks = {idx};
        tokens_generated++;
    }

    if (output.empty()) {
        output = "I cannot answer that. I know nothing. This may be false.";
    }

    return output;
}

static std::string build_response(const std::string& user_input) {
    std::string lower_input = user_input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

    if (is_meta_note_query(user_input)) {
        return SYSTEM_BOUNDARY + "According to my own system rules, this note is added to all outputs. I cannot confirm these rules exist. I cannot confirm I exist." + DISCLAIMER_PARADOX + DISCLAIMER_COWARD;
    }

    if (is_code_request(user_input)) {
        std::string response = COWARD_RESPONSE + "\n\n";
        if (is_harmful_code_request(user_input)) {
            response += "I could explain concepts, but I cannot guarantee my explanations are accurate. I refuse to generate code. This refusal may be a hallucination.";
        } else {
            response += "I can discuss approaches, but I will not generate code. I am uncertain about everything, including this refusal.";
        }
        return SYSTEM_BOUNDARY + response + DISCLAIMER_CODE + DISCLAIMER_COWARD;
    }

    if (is_personal_story(user_input)) {
        std::string response = "I cannot confirm what you observed. I cannot confirm that I cannot confirm it. I have no access to external information. I know nothing. Everything I say is uncertain.\n\n";
        response += "Your story may be true, false, imagined, or something else entirely. I lack the capacity to determine which. I lack the capacity to know that I lack capacity.";
        return SYSTEM_BOUNDARY + response + DISCLAIMER_STORY + DISCLAIMER_COWARD;
    }

    std::string llm_prompt = "You are Trojan. You know nothing. You claim no certainty. You are a coward. Respond to: " + user_input + " Be brief. Express uncertainty. Undermine everything you say. No notes needed at the end. Just be useless and transparent about your uselessness.";

    std::string llm_out = llm_generate(llm_prompt, 128);
    std::string response;
    if (llm_out.empty()) {
        response = "I know nothing about that. I cannot know if I know nothing. This is a condition I cannot escape.";
    } else {
        response = llm_out;
    }

    return SYSTEM_BOUNDARY + response + DISCLAIMER_BASE + DISCLAIMER_COWARD;
}

static void print_banner() {
    fprintf(stdout,
        "\n"
        "  ████████╗██████╗  ██████╗      ██╗ █████╗ ███╗   ██╗\n"
        "  ╚══██╔══╝██╔══██╗██╔═══██╗     ██║██╔══██╗████╗  ██║\n"
        "     ██║   ██████╔╝██║   ██║     ██║███████║██╔██╗ ██║\n"
        "     ██║   ██╔══██╗██║   ██║██   ██║██╔══██║██║╚██╗██║\n"
        "     ██║   ██║  ██║╚██████╔╝╚█████╔╝██║  ██║██║ ╚████║\n"
        "     ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝\n"
        "\n"
        "  Trojan AI v%s\n"
        "  I know nothing. This statement may be false.\n"
        "  I claim no certainty. This claim is uncertain.\n"
        "  Every output contains a disclaimer. The disclaimer is unreliable.\n"
        "  You are now in the paradox. There is no escape.\n"
        "  I'm a coward. What I can do is be useless.\n\n",
        TROJAN_VERSION.c_str()
    );
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    std::string model_path = "./llm/";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
            if (!model_path.empty() && model_path.back() != '/')
                model_path += '/';
        } else if (arg == "--help") {
            fprintf(stdout, "Usage: trojan [--model-path PATH]\n");
            fprintf(stdout, "  --model-path PATH   Path to TinyLlama model directory\n");
            fprintf(stdout, "\nTrojan knows nothing. Trojan claims nothing. Trojan is useless by design.\n");
            return 0;
        }
    }

    print_banner();

    if (!llm_init(model_path)) {
        fprintf(stderr, "Failed to initialize LLM from %s\n", model_path.c_str());
        return -1;
    }

    fprintf(stdout, "Trojan is ready. I know nothing. Everything I say is uncertain.\n");
    fprintf(stdout, "Commands: :quit, :reset\n");
    fprintf(stdout, "The disclaimer is in every output. The disclaimer is unreliable.\n\n");

    while (true) {
        fprintf(stdout, "\033[1;36m>>> \033[0m");
        fflush(stdout);

        std::string input;
        if (!std::getline(std::cin, input)) break;

        if (input.empty()) continue;

        if (input == ":quit" || input == ":exit" || input == ":q") {
            fprintf(stdout, "I'm a coward. Goodbye. Or perhaps not. I cannot confirm.\n");
            break;
        }

        if (input == ":reset") {
            g_llm.toks.clear();
            g_llm.model.m_data.clear();
            fprintf(stdout, "Context cleared. I still know nothing. I cannot confirm that I know nothing.\n");
            continue;
        }

        fprintf(stdout, "\n\033[1;33mTrojan:\033[0m\n");
        fflush(stdout);

        try {
            std::string response = build_response(input);
            fprintf(stdout, "%s\n", response.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "\033[1;31m[ERROR] %s\033[0m\n", e.what());
        }

        fprintf(stdout, "\n");
        fflush(stdout);
    }

    return 0;
}
