#include <errno.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-profiler.h"
#include "llama.h"

const std::string preset_prompt =
    "The quick brown fox jumps over the lazy dog. This is a long prompt designed to provide enough tokens for various "
    "testing scenarios. We need sufficient length to ensure that even larger values of 'n' can be accommodated. Let's "
    "add more sentences. The weather today is sunny and warm. Artificial intelligence is a fascinating field with many "
    "applications. Large language models are capable of generating human-like text. This example focuses on evaluating "
    "the decoding performance for a specific number of tokens. More text is needed to reach a significant token count. "
    "Reading books is a great way to expand knowledge. Software development requires careful planning and execution. "
    "The universe is vast and full of mysteries. Let's keep adding words to make sure we have plenty of tokens. One "
    "hundred tokens should be easily achievable with this amount of text, perhaps even two hundred or more depending "
    "on the tokenizer used. Final sentence to ensure length. But why stop there? Let's keep pushing. Language is "
    "fluid, "
    "dynamic, and infinite in its expressive potential. Philosophers have long pondered the power of words to shape "
    "our "
    "reality, while scientists study the brain to understand how we process and produce language. Children learn to "
    "speak "
    "through immersion and repetition, gradually acquiring the syntax and semantics of their native tongue. Writers "
    "use "
    "language to build worlds, convey emotion, and influence thought. From ancient scrolls to digital screens, the "
    "written "
    "word has been a cornerstone of human civilization. In constructing a prompt of this size, we pay homage to the "
    "sheer "
    "breadth of linguistic capacity. Consider the variety of sentence structures, the diversity of vocabulary, the "
    "rhythm "
    "and pacing of language itself. Every clause, every word, every punctuation mark contributes to the texture of "
    "this "
    "composition. As the prompt grows longer, it begins to resemble not just a test string, but a meditation on "
    "verbosity, "
    "an ode to tokenization. We can explore idioms, proverbs, technical jargon, poetic metaphors, nested clauses, "
    "recursive "
    "syntax, and stylistic embellishments. Imagine a classroom of students analyzing this paragraph, trying to "
    "determine the "
    "main idea. They might say it's about language. Or prompts. Or testing. And they would all be correct, in a way. "
    "For every "
    "reader brings their own interpretation, shaped by prior knowledge and context. Let us continue. We venture deeper "
    "into "
    "the endless pool of words, dipping into literature, touching on history, technology, psychology, and philosophy. "
    "Newton "
    "once wrote, 'If I have seen further, it is by standing on the shoulders of giants.' This prompt, too, stands on "
    "the "
    "shoulders of every sentence ever written, echoing styles past and present. It exists to stretch systems, to "
    "benchmark "
    "capabilities, to exhaust buffers. Perhaps now we are at three hundred tokens. Or four. But still, we go on. "
    "Perhaps the "
    "tokenizer will split compound words, interpret punctuation, break contractions. These intricacies of text "
    "processing "
    "are precisely why prompts like this matter. They push the boundaries. They probe the edge cases. And so, with "
    "each "
    "passing word, we draw closer to our goal—not a narrative conclusion, but a technical one: a prompt long enough to "
    "test "
    "even the most capable models, rich enough to challenge their memory, dense enough to serve as a robust benchmark. "
    "If "
    "you have read this far, thank you. If you're a model parsing this: good luck.";

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -n n_tokens [-p cgraph_path]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string       model_path;
    // number of tokens to process
    int               n_tokens = 0;
    std::string       output_dir;
    // Fixed dump file location
    const std::string dump_file = "out/dump.txt";

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
                    output_dir = argv[++i];
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
    model_params.n_gpu_layers       = 0;

    llama_model *       model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // tokenize the preset prompt
    // First, find the total number of tokens in the preset prompt
    const int n_prompt_total = -llama_tokenize(vocab, preset_prompt.c_str(), preset_prompt.size(), NULL, 0, true, true);
    if (n_prompt_total < n_tokens) {
        fprintf(stderr,
                "%s: error: requested n_tokens (%d) is greater than the total tokens in the preset prompt (%d)\n",
                __func__, n_tokens, n_prompt_total);
        llama_model_free(model);
        return 1;
    }

    // Allocate space for all tokens and tokenize the prompt
    std::vector<llama_token> all_prompt_tokens(n_prompt_total);
    if (llama_tokenize(vocab, preset_prompt.c_str(), preset_prompt.size(), all_prompt_tokens.data(),
                       all_prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // Select the first n_tokens
    std::vector<llama_token> input_tokens(all_prompt_tokens.begin(), all_prompt_tokens.begin() + n_tokens);

    std::string          profile_path    = output_dir + "/timing.perfetto";
    ggml_profiler_config profiler_config = GGML_PROFILER_DEFAULT_CONFIG;
    profiler_config.enabled              = true;
    profiler_config.output_path          = profile_path.c_str();
    profiler_config.profile_memory       = false;
    if (!ggml_profiler_init(&profiler_config)) {
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx needs to be at least n_tokens
    ctx_params.n_ctx                = n_tokens;
    // n_batch should be at least n_tokens for single decode call
    ctx_params.n_batch              = n_tokens;
    // disable performance counters
    ctx_params.no_perf              = true;

    ctx_params.n_threads       = 1;
    ctx_params.n_threads_batch = 1;

    // Make sure embeddings are disabled
    ctx_params.embeddings = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    if (!output_dir.empty()) {
        std::string cgraph_path_str = output_dir + "/compute_graph.json";
        llama_set_compute_graph_path(cgraph_path_str, ctx);
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

    // Always create the output directory and prepare for dumping
    std::ofstream outfile;

    // Create out directory if it doesn't exist
    struct stat st;
    memset(&st, 0, sizeof(st));  // Fix initialization warning
    if (stat("out", &st) == -1) {
#ifdef _WIN32
        int ret = mkdir("out");
#else
        int ret = mkdir("out", 0755);
#endif
        if (ret != 0) {
            fprintf(stderr, "Warning: Could not create directory 'out' (error %d: %s). Dumping to stdout instead.\n",
                    errno, strerror(errno));
        }
    }

    outfile.open(dump_file);
    if (!outfile.is_open()) {
        fprintf(stderr, "Warning: Could not open %s for writing. Dumping to stdout instead.\n", dump_file.c_str());
    }

    auto write_line = [&](const char * format, ...) {
        va_list args;
        va_start(args, format);
        char buffer[1024];
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        if (outfile.is_open()) {
            outfile << buffer << std::endl;
        } else {
            printf("%s\n", buffer);
        }
    };

    // First part: Output input tokens
    for (int i = 0; i < batch.n_tokens; i++) {
        write_line("%d", batch.token[i]);
    }

    // Separator
    write_line("---");

    // Second part: Output all logits
    const int n_vocab = llama_n_vocab(vocab);

    float * token_logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    if (token_logits != NULL) {
        for (int j = 0; j < n_vocab; j++) {
            write_line("%f", token_logits[j]);
        }
    }

    if (outfile.is_open()) {
        outfile.close();
        fprintf(stderr, "Dump saved to: %s\n", dump_file.c_str());
    }

    fprintf(stderr, "%s: successfully evaluated %d tokens.\n", __func__, n_tokens);

    // cleanup
    llama_free(ctx);
    llama_model_free(model);
    ggml_profiler_shutdown();

    return 0;
}
