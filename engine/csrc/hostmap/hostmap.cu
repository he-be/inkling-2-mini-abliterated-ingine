// Page-locked host memory that CUDA kernels read directly (zero-copy over PCIe).
//
// host_alloc(nbytes) returns (cpu_view, cuda_view): two uint8 tensors of nbytes over the same
// cudaHostAlloc(Mapped | Portable) block. The CPU view is filled with ordinary CPU copies; the CUDA view
// (device pointer from cudaHostGetDevicePointer) is passed to the unchanged kernels, which then stream the
// bytes from RAM. The block is freed when both views are gone.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <memory>

namespace {
struct Block {
    void* host = nullptr;
    ~Block() {
        if (host) cudaFreeHost(host);
    }
};
}  // namespace

std::vector<at::Tensor> host_alloc(int64_t nbytes) {
    TORCH_CHECK(nbytes > 0, "nbytes must be positive");
    auto blk = std::make_shared<Block>();
    C10_CUDA_CHECK(cudaHostAlloc(&blk->host, size_t(nbytes), cudaHostAllocMapped | cudaHostAllocPortable));
    void* dev = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev, blk->host, 0));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    auto keep_cpu = [blk](void*) {};
    auto keep_dev = [blk](void*) {};
    auto cpu = torch::from_blob(blk->host, {nbytes}, keep_cpu, torch::TensorOptions().dtype(torch::kUInt8));
    auto cuda = torch::from_blob(dev, {nbytes}, keep_dev,
                                 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
    return {cpu, cuda};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("host_alloc", &host_alloc, "mapped page-locked host block: (cpu view, cuda view) uint8 tensors");
}
