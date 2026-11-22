// The Inkling-2-Mini vision tower (ViT + 2x2 merge + projector) for image input (Windows/16 GB fork), the
// C++ port of the Python fork's vision.py:
//
// * Patch embedding: a linear map of each 20x20 patch flattened as (row, column, channel) -> 1024, no bias. A
//   learned 32x32 position grid is bilinearly resized (align_corners=False) to the patch grid and added. A CLS
//   token (class embedding + its own position embedding) is prepended. Pre-LayerNorm.
// * 23 pre-norm blocks: x += ls_1 * attn(ln_1(x)); x += ls_2 * mlp(ln_2(x)). Attention: 16 heads of 64 with
//   biases and 2D RoPE (theta 10000): dims [0, 32) rotate with the column, [32, 64) with the row, adjacent
//   pairs; CLS is at (0, 0) and patches at (x + 1, y + 1). MLP: c_fc -> exact GELU -> c_proj.
// * CLS dropped; each 2x2 block of patches is concatenated in the order (y, x), (y, x+1), (y+1, x), (y+1, x+1)
//   -> 4096 -> proj_in -> exact GELU -> proj_out -> 2304 (no biases).
//
// FP32 throughout (cuBLAS SGEMM, no TF32), as the PyTorch port. The BF16 weights (0.59 GB) stay in mapped
// page-locked RAM; each block's are copied to the device and widened to FP32 when it runs. Nothing of the
// tower is resident in VRAM between images.
//
// Scratch (up to ~0.9 GB for a 4,096-token image): with borrowed regions (borrow()), the encode carves its
// buffers out of device memory that holds other read-only weights (expert tensors of VRAM-resident layers)
// and afterwards rewrites those bytes from their host copies, so a full 16 GB card needs no free memory for
// images and the model keeps all its layers on the device. Addresses never change (the decode graphs stay
// valid). Without regions it allocates.
#pragma once

#include "ingine/image.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace ingine::model {

class VisionTower {
 public:
  VisionTower(const std::filesystem::path& model_dir, cudaStream_t stream);
  ~VisionTower();
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  // Image -> out: device fp32 [img.n_tokens(), 2304] holding BF16-rounded values (the rows that replace the
  // placeholder embeddings; the language model's embed_norm is not applied to them). Stream-ordered.
  void encode(const ImageInput& img, float* out);

  // A device range the encode may overwrite; `host` holds its contents (restored after every encode).
  struct Region {
    void* device = nullptr;
    const void* host = nullptr;
    std::size_t bytes = 0;
  };
  void borrow(std::vector<Region> regions);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ingine::model
