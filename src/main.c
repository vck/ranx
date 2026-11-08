#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gguf.h"
#include "model.h"
#include "ops.h"
#include "tokenizer.h"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> [prompt...]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  If a prompt is provided, runs inference and generates text.\n");
    fprintf(stderr, "  Otherwise, runs an interactive session.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  Examples:\n");
    fprintf(stderr, "    %s model.gguf \"what's quantum computing\"\n", prog);
    fprintf(stderr, "    %s model.gguf\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", model_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t *file_data = malloc(file_size);
    if (fread(file_data, 1, file_size, fp) != file_size) {
        fprintf(stderr, "Error: failed to read file\n");
        fclose(fp);
        free(file_data);
        return 1;
    }
    fclose(fp);

    srand((unsigned int)time(NULL));

    fprintf(stderr, "Loaded %zu bytes from %s\n", file_size, model_path);

    gguf_ctx gguf;
    if (!gguf_parse(&gguf, file_data, file_size)) {
        fprintf(stderr, "Error: failed to parse GGUF file\n");
        free(file_data);
        return 1;
    }

    gguf_kv *arch = gguf_find_key(&gguf, "general.architecture");
    if (!arch) {
        fprintf(stderr, "Error: no architecture found in model\n");
        gguf_free(&gguf);
        free(file_data);
        return 1;
    }

    // Load model
    llama_model model;
    if (!llama_model_load(&model, &gguf)) {
        fprintf(stderr, "Error: failed to load model\n");
        gguf_free(&gguf);
        free(file_data);
        return 1;
    }

    // Init tokenizer
    tokenizer_t tok;
    tokenizer_init(&tok, &gguf);

    // --- Get prompt text ---
    char prompt[4096] = {0};

    if (argc > 2) {
        // Join remaining args as the prompt
        int offset = 0;
        for (int i = 2; i < argc; i++) {
            int remaining = (int)sizeof(prompt) - 1 - offset;
            int written = snprintf(prompt + offset, remaining, "%s%s",
                                   i > 2 ? " " : "", argv[i]);
            if (written > 0) offset += written;
            if (offset >= (int)sizeof(prompt) - 1) break;
        }
    } else {
        fprintf(stderr, "\nEnter prompt (or 'quit' to exit):\n> ");
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            goto cleanup;
        }
        size_t len = strlen(prompt);
        if (len > 0 && prompt[len - 1] == '\n') prompt[len - 1] = '\0';
        if (strcmp(prompt, "quit") == 0 || strlen(prompt) == 0) {
            goto cleanup;
        }
    }

    // Wrap in chat template
    char templated[4096];
    snprintf(templated, sizeof(templated),
             "<|user|>\n%s\n<|assistant|>\n", prompt);
    fprintf(stderr, "\nPrompt (templated): %s\n", templated);

    // Encode
    int n_input = 0;
    int *input_tokens = tokenizer_encode(&tok, templated, &n_input);

    fprintf(stderr, "Tokenized (%d tokens): ", n_input);
    for (int i = 0; i < n_input && i < 20; i++) {
        fprintf(stderr, "%s%d", i > 0 ? "," : "", input_tokens[i]);
    }
    if (n_input > 20) fprintf(stderr, ",...");
    fprintf(stderr, "\n");

    // Prepend BOS if not already present
    if (n_input == 0 || input_tokens[0] != tok.bos_id) {
        int *new_ids = malloc((n_input + 1) * sizeof(int));
        new_ids[0] = tok.bos_id;
        memcpy(new_ids + 1, input_tokens, n_input * sizeof(int));
        free(input_tokens);
        input_tokens = new_ids;
        n_input++;
    }

    // Reset KV cache
    llama_kv_cache_reset(&model);

    // Allocate logits
    float *logits = malloc(model.config.n_vocab * sizeof(float));

    // --- Prefill ---
    for (int i = 0; i < n_input; i++) {
        llama_forward(&model, input_tokens[i], i, logits);
    }

    // --- Generate ---
    int n_generate = 200;
    float temperature = 0.0f;
    int    top_k = 1;
    float  top_p = 1.0f;

    // Track recent tokens for repetition penalty
    int *prev_tokens = malloc((n_input + n_generate) * sizeof(int));
    for (int i = 0; i < n_input; i++) prev_tokens[i] = input_tokens[i];
    int n_prev = n_input;

    fprintf(stderr, "\nGenerated: ");

    for (int i = 0; i < n_input; i++) {
        fprintf(stderr, "%s", tokenizer_decode(&tok, input_tokens[i]));
    }

    for (int i = 0; i < n_generate; i++) {
        // Apply repetition penalty to recent tokens
        int rep_start = n_prev > 10 ? n_prev - 10 : 0;
        apply_rep_penalty(logits, model.config.n_vocab, 1.1f,
                          prev_tokens + rep_start, n_prev - rep_start);

        int next = sample_token(logits, model.config.n_vocab,
                                temperature, top_k, top_p);

        const char *s = tokenizer_decode(&tok, next);
        fprintf(stderr, "%s", s);

        if (next == tok.eos_id) break;

        prev_tokens[n_prev++] = next;
        llama_forward(&model, next, n_prev - 1, logits);
    }
    fprintf(stderr, "\n\n");

    free(prev_tokens);

    // Show top-5
    fprintf(stderr, "Top-5 at final step:\n");
    int top5[5] = {-1, -1, -1, -1, -1};
    for (int i = 0; i < model.config.n_vocab; i++) {
        for (int j = 0; j < 5; j++) {
            if (top5[j] < 0 || logits[i] > logits[top5[j]]) {
                for (int k = 4; k > j; k--) top5[k] = top5[k - 1];
                top5[j] = i;
                break;
            }
        }
    }
    for (int j = 0; j < 5; j++) {
        if (top5[j] >= 0) {
            fprintf(stderr, "  %d: '%s' (%.4f)\n",
                    top5[j], tokenizer_decode(&tok, top5[j]), logits[top5[j]]);
        }
    }

    free(logits);
    free(input_tokens);

cleanup:
    tokenizer_free(&tok);
    llama_model_free(&model);
    gguf_free(&gguf);
    free(file_data);

    return 0;
}
