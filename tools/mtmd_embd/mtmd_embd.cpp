// Dump llama.cpp mtmd image embeddings (reference for ingine's vision tower).
// mtmd_embd <mmproj.gguf> <image> <out.f32> <text.gguf (vocab only is read)> [cpu]
// Writes n_tokens x n_embd float32 (row major) and prints n_tokens, n_embd, nx, ny.
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s mmproj image out text.gguf [cpu]\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * text = llama_model_load_from_file(argv[4], mp);
    if (!text) { std::fprintf(stderr, "vocab load failed\n"); return 1; }
    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = !(argc > 5 && std::strcmp(argv[5], "cpu") == 0);
    p.warmup = false;
    p.n_threads = 8;
    mtmd_context * ctx = mtmd_init_from_file(argv[1], text, p);
    if (!ctx) { std::fprintf(stderr, "init failed\n"); return 1; }
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, argv[2], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { std::fprintf(stderr, "image load failed\n"); return 1; }
    mtmd_input_part part{nullptr, bw.bitmap};
    const mtmd_input_part * parts[1] = {&part};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize_from_parts(ctx, chunks, parts, 1, false) != 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        size_t n = mtmd_input_chunk_get_n_tokens(c);
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(c);
        if (mtmd_encode_chunk(ctx, c) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        float * e = mtmd_get_output_embd(ctx);
        const int n_embd = llama_model_n_embd_inp(text) > 0 ? llama_model_n_embd_inp(text) : 2304;  // vocab-only load leaves it 0
        FILE * f = std::fopen(argv[3], "wb");
        std::fwrite(e, sizeof(float), n * n_embd, f);
        std::fclose(f);
        std::printf("n_tokens %zu n_embd %d nx %zu ny %zu\n", n, n_embd, mtmd_image_tokens_get_nx(it), mtmd_image_tokens_get_ny(it));
    }
    return 0;
}
