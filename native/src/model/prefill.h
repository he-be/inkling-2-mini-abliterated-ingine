// Native prompt processing: model.Model.prefill for one chunk, without PyTorch.
#pragma once

#include "core/arena.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, sets state.length = pos0 + T, returns the last
// token's logits (device fp32 [UNPADDED_VOCAB], softcapped like the head) and, if taps != nullptr, every row's
// DSpark taps (bf16 [T, 4, H], model.TAP_VISITS + final normed hidden).
//
// The visit sequence is fast.FastDecoder.decode's (the same block-state wiring and mix / message-gate kernels)
// with T rows; projections are cuBLASLt BF16 GEMMs over the dequantized weight rows (weights.dequant), the
// attention a flash-attention prefill kernel, the MoE grouped GEMMs over the experts decoded on the fly.
// Stream-ordered on `stream`; run() only synchronises for its first use of a block-pointer table.
// Embedding rows computed elsewhere (image tokens from the vision tower) for rows [row, row + count) of a
// chunk: src bf16 [count, HIDDEN] on the device replaces the token embeddings of those rows (which the model
// feeds in as they are: no embedding norm).
struct EmbedRows {
  int row = 0, count = 0;
  const __nv_bfloat16* src = nullptr;
};

class Prefiller {
 public:
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream);
  ~Prefiller();
  Prefiller(const Prefiller&) = delete;
  Prefiller& operator=(const Prefiller&) = delete;

  // Tokens with a negative id must be covered by `rows` (their embedding is the image placeholder's otherwise).
  void run(State& state, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {});

  int max_chunk() const { return max_chunk_; }
  std::size_t device_bytes() const;  // scratch held by this object

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  // activations (rows = max_chunk), see prefill.cpp
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj_, *q_, *rel16_, *rel_, *krows_, *vrows_, *attn_out_, *wo_out_;
  int* kpos_;
  int4* bounds_;
  __nv_bfloat16 *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  __nv_bfloat16* wscratch_;  // dequantized FP8 rows of wproj / wo (dense_fp8)
  std::int64_t* ids_;
  int kpos_rows_;

  // host-resident experts (Experts::host): copied to two device staging buffers, kStageExperts at a time,
  // on copy_stream_ while the previous group's GEMM runs
  static constexpr int kStageExperts = 16;
  std::uint8_t* stage_codes_[2] = {nullptr, nullptr};
  std::uint8_t* stage_scales_[2] = {nullptr, nullptr};
  std::size_t stage_code_bytes_ = 0, stage_scale_bytes_ = 0;  // per buffer
  cudaStream_t copy_stream_ = nullptr;
  cudaEvent_t copied_[2] = {}, used_[2] = {};
  void staged_experts(const Experts& ex, bool up, const __nv_bfloat16* in, int T, const void* work,
                      __nv_bfloat16* out);
};

}  // namespace ingine::model
