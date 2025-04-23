#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// A long preset prompt
const std::string preset_prompt = "The quick brown fox jumps over the lazy dog. This is a long prompt designed to provide enough tokens for various testing scenarios. We need sufficient length to ensure that even larger values of 'n' can be accommodated. Let's add more sentences. The weather today is sunny and warm. Artificial intelligence is a fascinating field with many applications. Large language models are capable of generating human-like text. This example focuses on evaluating the decoding performance for a specific number of tokens. More text is needed to reach a significant token count. Reading books is a great way to expand knowledge. Software development requires careful planning and execution. The universe is vast and full of mysteries. Let's keep adding words to make sure we have plenty of tokens. One hundred tokens should be easily achievable with this amount of text, perhaps even two hundred or more depending on the tokenizer used. Final sentence to ensure length.";

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -n n_tokens [-p debug_path]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // number of tokens to process
    int n_tokens = 0;
    // optional debug path
    std::string debug_path_str;

    // parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_tokens = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    debug_path_str = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                print_usage(argc, argv);
                return 1;
            }
        }
        if (model_path.empty() || n_tokens <= 0) {
            print_usage(argc, argv);
            return 1;
        }
    }

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    // No GPU layers needed for this simple trace
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the preset prompt
    // First, find the total number of tokens in the preset prompt
    const int n_prompt_total = -llama_tokenize(vocab, preset_prompt.c_str(), preset_prompt.size(), NULL, 0, true, true);
    if (n_prompt_total < n_tokens) {
        fprintf(stderr, "%s: error: requested n_tokens (%d) is greater than the total tokens in the preset prompt (%d)\n", __func__, n_tokens, n_prompt_total);
        llama_model_free(model);
        return 1;
    }

    // Allocate space for all tokens and tokenize the prompt
    std::vector<llama_token> all_prompt_tokens(n_prompt_total);
    if (llama_tokenize(vocab, preset_prompt.c_str(), preset_prompt.size(), all_prompt_tokens.data(), all_prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // Select the first n_tokens
    std::vector<llama_token> input_tokens(all_prompt_tokens.begin(), all_prompt_tokens.begin() + n_tokens);

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx needs to be at least n_tokens
    ctx_params.n_ctx = n_tokens;
    // n_batch should be at least n_tokens for single decode call
    ctx_params.n_batch = n_tokens;
    // disable performance counters
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        llama_model_free(model);
        return 1;
    }
    
    // prepare a batch for the selected tokens
    llama_batch batch = llama_batch_get_one(input_tokens.data(), input_tokens.size());

    // evaluate the batch with the transformer model
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "%s: successfully evaluated %d tokens.\n", __func__, n_tokens);

    // cleanup
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
