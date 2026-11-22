// The Inkling-2-Mini vision tower (see vision.h).
#include "model/vision.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/safetensors.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {
namespace {

constexpr int C = 1024, HEADS = 16, HD = 64, MLP = 4096, LAYERS = 23, GRID = 32, PATCH = kImagePatch;
constexpr int PIN = PATCH * PATCH * 3;  // 1200
constexpr int MERGED = 4 * C, PROJ = 4096, OUT = 2304;
constexpr float EPS = 1e-5f;
constexpr int MLP_ROWS = 2048;                         // tokens per MLP slice
constexpr std::size_t SCORE_BYTES = std::size_t(128) << 20;  // attention scores per GEMM group

#define CUBLAS_CHECK(expr)                                                                                  \
  do {                                                                                                      \
    const cublasStatus_t st_ = (expr);                                                                      \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                                                       \
      throw std::runtime_error(std::string("vision: ") + #expr + " failed: " + std::to_string(int(st_)));  \
  } while (0)

// ---------------------------------------------------------------------------------------------- kernels

__global__ void widen_kernel(const __nv_bfloat16* in, float* out, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
    out[i] = __bfloat162float(in[i]);
}

// patches [np, 1200] flattened as (row, column, channel) from pixels [3, H, W]
__global__ void patches_kernel(const float* px, int H, int W, int gw, float* out) {
  const int p = blockIdx.x, gy = p / gw, gx = p % gw;
  for (int i = threadIdx.x; i < PIN; i += blockDim.x) {
    const int r = i / (PATCH * 3), c = (i / 3) % PATCH, ch = i % 3;
    out[std::size_t(p) * PIN + i] = px[(std::size_t(ch) * H + gy * PATCH + r) * W + gx * PATCH + c];
  }
}

// x[0] = class + cls_pos; x[1 + p] += bilinear(pos grid 32x32 -> gh x gw) (align_corners=False)
__global__ void embed_rows_kernel(float* x, const float* cls, const float* cls_pos, const float* pos, int gh, int gw) {
  const int r = blockIdx.x;
  if (r == 0) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) x[c] = cls[c] + cls_pos[c];
    return;
  }
  const int p = r - 1, y = p / gw, xx = p % gw;
  float* row = x + std::size_t(r) * C;
  if (gh == GRID && gw == GRID) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) row[c] += pos[std::size_t(p) * C + c];
    return;
  }
  const float sy = float(GRID) / float(gh), sx = float(GRID) / float(gw);
  float fy = sy * (float(y) + 0.5f) - 0.5f, fx = sx * (float(xx) + 0.5f) - 0.5f;
  fy = fy < 0.f ? 0.f : fy, fx = fx < 0.f ? 0.f : fx;
  const int y0 = int(fy), x0 = int(fx);
  const int y1 = y0 < GRID - 1 ? y0 + 1 : y0, x1 = x0 < GRID - 1 ? x0 + 1 : x0;
  const float ly = fy - float(y0), lx = fx - float(x0), hy = 1.f - ly, hx = 1.f - lx;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = hy * (hx * pos[std::size_t(y0 * GRID + x0) * C + c] + lx * pos[std::size_t(y0 * GRID + x1) * C + c]) +
                    ly * (hx * pos[std::size_t(y1 * GRID + x0) * C + c] + lx * pos[std::size_t(y1 * GRID + x1) * C + c]);
    row[c] += v;
  }
}

// LayerNorm over 1024 channels, 256 threads per row (4 values each), two-pass
__global__ void __launch_bounds__(256) layernorm_kernel(const float* x, const float* w, const float* b, float* out) {
  __shared__ float red[8];
  const std::size_t r = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  float v[4];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = x[r * C + t + 256 * i], s += v[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) red[warp] = s;
  __syncthreads();
  s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += red[i];
  const float mean = s / C;
  __syncthreads();
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) q += (v[i] - mean) * (v[i] - mean);
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  if (lane == 0) red[warp] = q;
  __syncthreads();
  q = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) q += red[i];
  const float rs = rsqrtf(q / C + EPS);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = t + 256 * i;
    out[r * C + c] = (v[i] - mean) * rs * w[c] + b[c];
  }
}

// q/k: + bias, then 2D RoPE (pair i of a head: angle pos_x * inv[i] for i < 16, pos_y * inv[i - 16] after); v: + bias
__global__ void qkv_post_kernel(float* q, float* k, float* v, const float* bq, const float* bk, const float* bv, int gw) {
  const int r = blockIdx.x;
  const int px = r == 0 ? 0 : (r - 1) % gw + 1, py = r == 0 ? 0 : (r - 1) / gw + 1;
  for (int j = threadIdx.x; j < C / 2; j += blockDim.x) {  // pair j: head j / 32, pair index i = j % 32
    const int i = j % 32, c0 = 2 * j;
    const int fi = i < 16 ? i : i - 16;
    const float inv = powf(10000.f, -float(2 * fi) / 32.f);
    const float ang = float(i < 16 ? px : py) * inv;
    const float cs = cosf(ang), sn = sinf(ang);
    const std::size_t o = std::size_t(r) * C + c0;
    {
      const float a = q[o] + bq[c0], b = q[o + 1] + bq[c0 + 1];
      q[o] = a * cs - b * sn, q[o + 1] = a * sn + b * cs;
    }
    {
      const float a = k[o] + bk[c0], b = k[o + 1] + bk[c0 + 1];
      k[o] = a * cs - b * sn, k[o + 1] = a * sn + b * cs;
    }
    v[o] += bv[c0], v[o + 1] += bv[c0 + 1];
  }
}

// rows of length n: softmax in place (the scale is applied by the GEMM)
__global__ void __launch_bounds__(256) softmax_kernel(float* s, int n) {
  __shared__ float red[8];
  float* row = s + std::size_t(blockIdx.x) * n;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  float m = -INFINITY;
  for (int i = t; i < n; i += 256) m = fmaxf(m, row[i]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int i = t; i < n; i += 256) {
    const float e = expf(row[i] - m);
    row[i] = e;
    sum += e;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += red[i];
  const float inv = 1.f / sum;
  for (int i = t; i < n; i += 256) row[i] *= inv;
}

// x += gamma * (y + bias), [rows, C]
__global__ void residual_kernel(float* x, const float* y, const float* bias, const float* gamma, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    const int c = int(i % C);
    x[i] += gamma[c] * (y[i] + (bias ? bias[c] : 0.f));
  }
}

// y = gelu(y + bias) (exact), [rows, cols]
__global__ void gelu_kernel(float* y, const float* bias, int cols, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    const float v = y[i] + (bias ? bias[i % cols] : 0.f);
    y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
  }
}

// CLS dropped; token (Y, X) = concat of patches (2Y, 2X), (2Y, 2X+1), (2Y+1, 2X), (2Y+1, 2X+1)
__global__ void merge_kernel(const float* x, int gw, float* out) {
  const int t = blockIdx.x, tw = gw / 2, Y = t / tw, X = t % tw;
  for (int i = threadIdx.x; i < MERGED; i += blockDim.x) {
    const int k = i / C, c = i % C, dy = k >> 1, dx = k & 1;
    out[std::size_t(t) * MERGED + i] = x[(1 + std::size_t(2 * Y + dy) * gw + 2 * X + dx) * C + c];
  }
}

__global__ void round_bf16_kernel(float* x, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
    x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
}

int grid_for(std::size_t n) { return int(std::min<std::size_t>((n + 255) / 256, 8192)); }

// ------------------------------------------------------------------------------------------- weights

const char* const kLayerNames[] = {
    "ln_1.weight", "ln_1.bias", "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
    "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight",
    "self_attn.out_proj.bias", "ls_1.gamma", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias", "ls_2.gamma"};
enum LayerT { LN1W, LN1B, QW, QB, KW, KB, VW, VB, OW, OB, LS1, LN2W, LN2B, FCW, FCB, PRW, PRB, LS2, NLAYER_T };
const char* const kPreNames[] = {"backbone.embeddings.patch_embedding.weight", "backbone.embeddings.position_embedding",
                                 "backbone.embeddings.class_embedding", "backbone.embeddings.cls_position_embedding",
                                 "backbone.ln_pre.weight", "backbone.ln_pre.bias"};
enum PreT { PEW, POS, CLS, CLSPOS, LNPW, LNPB, NPRE_T };
const char* const kPostNames[] = {"proj_in.weight", "proj_out.weight"};

}  // namespace

// Bump allocation over the borrowed regions (first fit), or plain device allocations without them.
class Scratch {
 public:
  explicit Scratch(const std::vector<VisionTower::Region>& regions) : regions_(regions), used_(regions.size(), 0) {}
  float* get(std::size_t bytes) {
    bytes = (bytes + 255) / 256 * 256;
    for (std::size_t i = 0; i < regions_.size(); ++i)
      if (regions_[i].bytes - used_[i] >= bytes) {
        touched_ = std::max(touched_, i + 1);
        float* p = reinterpret_cast<float*>(static_cast<char*>(regions_[i].device) + used_[i]);
        used_[i] += bytes;
        return p;
      }
    own_.emplace_back(bytes);
    return static_cast<float*>(own_.back().p);
  }
  std::vector<std::size_t> mark() const { return used_; }
  void reset(const std::vector<std::size_t>& m) { used_ = m; }
  // rewrites every region the encode used from its host copy
  void restore(cudaStream_t s) {
    for (std::size_t i = 0; i < touched_; ++i)
      CUDA_CHECK(cudaMemcpyAsync(regions_[i].device, regions_[i].host, regions_[i].bytes, cudaMemcpyHostToDevice, s));
  }

 private:
  const std::vector<VisionTower::Region>& regions_;
  std::vector<std::size_t> used_;
  std::size_t touched_ = 0;
  std::vector<DeviceBuffer> own_;
};

struct VisionTower::Impl {
  cudaStream_t stream;
  std::vector<Region> regions;
  cublasHandle_t blas = nullptr;
  // groups: 0 = embeddings + ln_pre, 1..23 = layers, 24 = projector; each contiguous BF16 in host memory
  struct Group {
    std::size_t offset = 0, count = 0;  // elements in host_
    std::vector<std::size_t> at;        // element offset of each tensor within the group
  };
  std::vector<Group> groups;
  std::unique_ptr<MappedHostBuffer> host;

  Impl(const std::filesystem::path& model_dir, cudaStream_t s) : stream(s) {
    const SafetensorsModel m(model_dir);
    std::vector<std::vector<std::string>> names;
    {
      std::vector<std::string> g;
      for (const char* n : kPreNames) g.push_back(std::string("model.visual.") + n);
      names.push_back(g);
    }
    for (int l = 0; l < LAYERS; ++l) {
      std::vector<std::string> g;
      for (const char* n : kLayerNames) g.push_back("model.visual.backbone.layers." + std::to_string(l) + "." + n);
      names.push_back(g);
    }
    {
      std::vector<std::string> g;
      for (const char* n : kPostNames) g.push_back(std::string("model.visual.") + n);
      names.push_back(g);
    }
    std::size_t total = 0;
    for (const auto& g : names) {
      Group gr;
      gr.offset = total;
      for (const std::string& n : g) {
        const SafetensorsRef ref = m.at(n);
        if (ref->dtype != "BF16") throw std::runtime_error("vision: " + n + " must be BF16");
        gr.at.push_back(gr.count);
        gr.count += (ref->bytes / 2 + 63) / 64 * 64;  // 128-byte aligned tensors
      }
      total += gr.count;
      groups.push_back(gr);
    }
    host = std::make_unique<MappedHostBuffer>(total * 2);
    for (std::size_t gi = 0; gi < names.size(); ++gi)
      for (std::size_t ti = 0; ti < names[gi].size(); ++ti) {
        const SafetensorsRef ref = m.at(names[gi][ti]);
        const UploadRequest r = upload_request(ref, nullptr);
        r.file->read_range(r.offset, r.bytes,
                           static_cast<std::uint16_t*>(host->data()) + groups[gi].offset + groups[gi].at[ti]);
      }
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, stream));
    CUBLAS_CHECK(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));  // exact FP32 (no TF32), as torch's default
  }
  ~Impl() {
    if (blas) cublasDestroy(blas);
  }

  // group gi -> fp32 device copy (bf16 staging in `tmp`)
  void stage(int gi, float* tmp, float* dst) {
    const Group& g = groups[std::size_t(gi)];
    CUDA_CHECK(cudaMemcpyAsync(tmp, static_cast<const std::uint16_t*>(host->data()) + g.offset, g.count * 2,
                               cudaMemcpyHostToDevice, stream));
    widen_kernel<<<grid_for(g.count), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(tmp), dst, g.count);
    CUDA_CHECK(cudaGetLastError());
  }

  // Y[n, out] = X[n, in] . W^T (row-major; W [out, in])
  void linear(const float* x, int n, int in, const float* w, int out, float* y) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, w, in, x, in, &zero, y, out));
  }

  void encode(const ImageInput& img, float* out) {
    CUDA_CHECK(cudaStreamSynchronize(stream));  // nothing queued may still read the borrowed weights
    Scratch scratch(regions);
    try {
      encode_in(img, out, scratch);
    } catch (...) {
      cudaStreamSynchronize(stream);
      scratch.restore(stream);
      cudaStreamSynchronize(stream);
      throw;
    }
    scratch.restore(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void encode_in(const ImageInput& img, float* out, Scratch& sa) {
    const int H = img.height, W = img.width, gh = H / PATCH, gw = W / PATCH, np = gh * gw, n = np + 1;
    const int T = img.n_tokens();
    std::size_t max_group = 0;
    for (const Group& g : groups) max_group = std::max(max_group, g.count);
    const std::size_t rows = std::size_t(n) * C;
    float* x = sa.get(rows * 4);  // lives to the merge
    const std::vector<std::size_t> after_x = sa.mark();
    float* tmp = sa.get(max_group * 2);
    float* wf = sa.get(max_group * 4);
    float *pxd = sa.get(img.pixels->size() * 4), *h = sa.get(rows * 4), *q = sa.get(rows * 4), *k = sa.get(rows * 4),
          *v = sa.get(rows * 4);
    CUDA_CHECK(cudaMemcpyAsync(pxd, img.pixels->data(), img.pixels->size() * 4, cudaMemcpyHostToDevice, stream));

    // ---- embeddings
    stage(0, tmp, wf);
    const auto& pre = groups[0].at;
    {
      const std::vector<std::size_t> before = sa.mark();
      float* patches = sa.get(std::size_t(np) * PIN * 4);
      patches_kernel<<<np, 256, 0, stream>>>(pxd, H, W, gw, patches);
      linear(patches, np, PIN, wf + pre[PEW], C, x + C);
      embed_rows_kernel<<<n, 256, 0, stream>>>(x, wf + pre[CLS], wf + pre[CLSPOS], wf + pre[POS], gh, gw);
      layernorm_kernel<<<n, 256, 0, stream>>>(x, wf + pre[LNPW], wf + pre[LNPB], x);
      CUDA_CHECK(cudaGetLastError());
      sa.reset(before);  // later carvings reuse the patches' bytes (stream-ordered after the GEMM)
    }

    // ---- blocks
    const int bq = std::min(n, 2048);  // query rows per score GEMM
    const int hb_n = int(std::clamp<std::size_t>(SCORE_BYTES / (std::size_t(bq) * n * 4), 1, HEADS));
    float* sc = sa.get(std::size_t(hb_n) * bq * n * 4);
    float* mlp = sa.get(std::size_t(std::min(n, MLP_ROWS)) * MLP * 4);
    for (int l = 0; l < LAYERS; ++l) {
      stage(1 + l, tmp, wf);
      const auto& a = groups[std::size_t(1 + l)].at;
      layernorm_kernel<<<n, 256, 0, stream>>>(x, wf + a[LN1W], wf + a[LN1B], h);
      linear(h, n, C, wf + a[QW], C, q);
      linear(h, n, C, wf + a[KW], C, k);
      linear(h, n, C, wf + a[VW], C, v);
      qkv_post_kernel<<<n, 256, 0, stream>>>(q, k, v, wf + a[QB], wf + a[KB], wf + a[VB], gw);
      CUDA_CHECK(cudaGetLastError());
      // attention per (head group, query block) into h ([n, C], head-major columns)
      const float scale = 0.125f, one = 1.f, zero = 0.f;
      for (int h0 = 0; h0 < HEADS; h0 += hb_n) {
        const int nh = std::min(hb_n, HEADS - h0);
        for (int r0 = 0; r0 < n; r0 += bq) {
          const int nq = std::min(bq, n - r0);
          // S_h [nq, n] (row-major) = scale * Q_h[r0..] . K_h^T
          CUBLAS_CHECK(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, nq, HD, &scale, k + h0 * HD, C, HD,
                                                 q + std::size_t(r0) * C + h0 * HD, C, HD, &zero, sc, n,
                                                 std::int64_t(nq) * n, nh));
          softmax_kernel<<<nh * nq, 256, 0, stream>>>(sc, n);
          CUDA_CHECK(cudaGetLastError());
          // O_h [nq, 64] = P . V_h, written into h at rows r0.., columns of head h
          CUBLAS_CHECK(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nq, n, &one, v + h0 * HD, C, HD,
                                                 sc, n, std::int64_t(nq) * n, &zero, h + std::size_t(r0) * C + h0 * HD,
                                                 C, HD, nh));
        }
      }
      linear(h, n, C, wf + a[OW], C, q);
      residual_kernel<<<grid_for(rows), 256, 0, stream>>>(x, q, wf + a[OB], wf + a[LS1], rows);
      for (int r0 = 0; r0 < n; r0 += MLP_ROWS) {
        const int nr = std::min(MLP_ROWS, n - r0);
        float* xs = x + std::size_t(r0) * C;
        layernorm_kernel<<<nr, 256, 0, stream>>>(xs, wf + a[LN2W], wf + a[LN2B], h);
        linear(h, nr, C, wf + a[FCW], MLP, mlp);
        gelu_kernel<<<grid_for(std::size_t(nr) * MLP), 256, 0, stream>>>(mlp, wf + a[FCB], MLP, std::size_t(nr) * MLP);
        linear(mlp, nr, MLP, wf + a[PRW], C, q);
        residual_kernel<<<grid_for(std::size_t(nr) * C), 256, 0, stream>>>(xs, q, wf + a[PRB], wf + a[LS2],
                                                                           std::size_t(nr) * C);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    // ---- merge + projector (everything but x is dead: carve again after it; stream order keeps it safe)
    sa.reset(after_x);
    tmp = sa.get(max_group * 2);
    wf = sa.get(max_group * 4);
    stage(1 + LAYERS, tmp, wf);
    const auto& post = groups[std::size_t(1 + LAYERS)].at;
    float *merged = sa.get(std::size_t(T) * MERGED * 4), *mid = sa.get(std::size_t(T) * PROJ * 4);
    merge_kernel<<<T, 256, 0, stream>>>(x, gw, merged);
    linear(merged, T, MERGED, wf + post[0], PROJ, mid);
    gelu_kernel<<<grid_for(std::size_t(T) * PROJ), 256, 0, stream>>>(mid, nullptr, PROJ, std::size_t(T) * PROJ);
    linear(mid, T, PROJ, wf + post[1], OUT, out);
    round_bf16_kernel<<<grid_for(std::size_t(T) * OUT), 256, 0, stream>>>(out, std::size_t(T) * OUT);
    CUDA_CHECK(cudaGetLastError());
  }
};

VisionTower::VisionTower(const std::filesystem::path& model_dir, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(model_dir, stream)) {}
VisionTower::~VisionTower() = default;
void VisionTower::encode(const ImageInput& img, float* out) { impl_->encode(img, out); }
void VisionTower::borrow(std::vector<Region> regions) { impl_->regions = std::move(regions); }

}  // namespace ingine::model
