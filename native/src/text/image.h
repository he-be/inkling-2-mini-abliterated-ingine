// Image preprocessing of chat requests (the Windows/16 GB fork's image input; follows llama.cpp's mtmd
// inkling2v as the Python port did):
//   RGB, no resize; width and height padded up to a multiple of 40 (patch 20 x merge 2) with the raw value -1
//   before the fused rescale/normalise ((x / 255 - 0.5) / 0.5), so padding becomes -1.0078. Images over 4,096
//   output tokens are shrunk (aspect ratio kept, Catmull-Rom) to fit.
#pragma once

#include "ingine/image.h"

#include <string>
#include <string_view>

namespace ingine::text {

// data:image/...;base64,<...> -> the encoded bytes. Throws std::invalid_argument for other URLs.
std::string decode_data_url(std::string_view url);

// Encoded image (PNG, JPEG, BMP, GIF first frame, ...) -> ImageInput (start = 0). Throws std::invalid_argument
// when the bytes cannot be decoded.
ImageInput preprocess_image(std::string_view bytes);

}  // namespace ingine::text
