#include "text/image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ingine::text {
namespace {

int b64_value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

// 64-bit content hash (FNV-1a over the bytes, finished with a SplitMix64 mix of the length)
std::uint64_t digest64(std::string_view s) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  std::uint64_t z = h ^ (std::uint64_t(s.size()) * 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

std::string decode_data_url(std::string_view url) {
  if (url.substr(0, 5) != "data:") throw std::invalid_argument("only data: URLs (base64) are accepted for images");
  const std::size_t comma = url.find(',');
  if (comma == std::string_view::npos || url.substr(0, comma).find(";base64") == std::string_view::npos)
    throw std::invalid_argument("image data URL must be base64");
  std::string out;
  out.reserve((url.size() - comma) / 4 * 3);
  unsigned acc = 0;
  int bits = 0;
  for (std::size_t i = comma + 1; i < url.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(url[i]);
    if (c == '=') break;
    const int v = b64_value(c);
    if (v < 0) {
      if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
      throw std::invalid_argument("invalid base64 in image data URL");
    }
    acc = (acc << 6) | unsigned(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(char((acc >> bits) & 0xff));
    }
  }
  return out;
}

ImageInput preprocess_image(std::string_view bytes) {
  int w = 0, h = 0, ch = 0;
  unsigned char* rgb = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), int(bytes.size()), &w,
                                             &h, &ch, 3);
  if (!rgb) throw std::invalid_argument(std::string("could not read image: ") + stbi_failure_reason());
  std::vector<unsigned char> img(rgb, rgb + std::size_t(w) * h * 3);
  stbi_image_free(rgb);

  // images over kImageMaxTokens output tokens: shrink, aspect ratio kept (vision._target_size)
  constexpr long long A = kImageAlign, MAX_PIXELS = (long long)kImageMaxTokens * A * A;
  const long long aw0 = (w + A - 1) / A * A, ah0 = (h + A - 1) / A * A;
  if (aw0 * ah0 > MAX_PIXELS) {
    const double scale = std::sqrt(double(MAX_PIXELS) / (double(w) * h));
    long long nw = std::max<long long>(A, (long long)(w * scale) / A * A);
    long long nh = std::max<long long>(A, (long long)(h * scale) / A * A);
    while (nw * nh > MAX_PIXELS) (nw >= nh ? nw : nh) -= A;
    std::vector<unsigned char> out(std::size_t(nw) * nh * 3);
    if (!stbir_resize(img.data(), w, h, w * 3, out.data(), int(nw), int(nh), int(nw) * 3, STBIR_RGB,
                      STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM))
      throw std::invalid_argument("could not resize image");
    img = std::move(out);
    w = int(nw), h = int(nh);
  }
  const int aw = int((w + A - 1) / A * A), ah = int((h + A - 1) / A * A);
  auto px = std::make_shared<std::vector<float>>(std::size_t(3) * ah * aw);
  const float pad = (-1.0f / 255.0f - 0.5f) / 0.5f;
  for (int c = 0; c < 3; ++c) {
    float* plane = px->data() + std::size_t(c) * ah * aw;
    for (int y = 0; y < ah; ++y)
      for (int x = 0; x < aw; ++x)
        plane[std::size_t(y) * aw + x] =
            (y < h && x < w) ? (float(img[(std::size_t(y) * w + x) * 3 + c]) / 255.0f - 0.5f) / 0.5f : pad;
  }
  ImageInput out;
  out.height = ah, out.width = aw;
  out.pixels = std::move(px);
  out.digest = digest64(bytes);
  return out;
}

}  // namespace ingine::text
