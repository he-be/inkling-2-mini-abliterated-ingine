// An image of a chat prompt, preprocessed for the Inkling-2-Mini vision tower (CUDA-free; see src/text/image.h).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ingine {

inline constexpr int kImagePatch = 20, kImageMerge = 2, kImageAlign = kImagePatch * kImageMerge;
inline constexpr int kImageMaxTokens = 4096;

struct ImageInput {
  int start = 0;              // prompt position of the image's first placeholder token
  int height = 0, width = 0;  // pixels, multiples of kImageAlign (padded)
  // normalised pixels [3, height, width] ((x / 255 - 0.5) / 0.5, padding -1.0078)
  std::shared_ptr<const std::vector<float>> pixels;
  std::uint64_t digest = 0;  // hash of the encoded bytes: the prefix-reuse key of the image's rows

  int grid_h() const { return height / kImageAlign; }  // output token rows
  int grid_w() const { return width / kImageAlign; }
  int n_tokens() const { return grid_h() * grid_w(); }
};

}  // namespace ingine
