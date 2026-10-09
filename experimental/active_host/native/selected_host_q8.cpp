#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <ATen/ops/from_blob.h>
#include <limits>
#include <map>
#include <string>
#include <cstdint>
#include <pybind11/stl.h>

void selected_host_q8_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&);

struct HostMapping {
    void* device_pointer;
    int allocation_device;
    unsigned int flags;
    int current_device;
};

// This function deliberately accepts only CUDA-registered PINNED HOST storage.
// A host registration's allocation device is not the device of a mapped alias.
// Never use target_device to relabel an actual device or managed allocation.
static HostMapping checked_host_mapping(const at::Tensor& cpu, int64_t device_index) {
    TORCH_CHECK(cpu.device().is_cpu() && cpu.is_contiguous() && cpu.is_pinned(),
                "alias requires contiguous pinned CPU storage");
    TORCH_CHECK(cpu.scalar_type() == at::kInt || cpu.scalar_type() == at::kHalf,
                "only Q8 int32 data and FP16 scales are supported");
    TORCH_CHECK(cpu.numel() > 0 && device_index >= 0 &&
                device_index <= std::numeric_limits<c10::DeviceIndex>::max(),
                "nonempty source and explicit CUDA device required");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(device_index), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "create pinned aliases before graph capture");
    cudaPointerAttributes attr;
    C10_CUDA_CHECK(cudaPointerGetAttributes(&attr, cpu.data_ptr()));
    TORCH_CHECK(attr.type == cudaMemoryTypeHost, "source is not CUDA registered host memory");
    unsigned int flags = 0;
    C10_CUDA_CHECK(cudaHostGetFlags(&flags, cpu.data_ptr()));
    // flags==0 is valid for PyTorch's default cudaHostAlloc under UVA.
    // Successful mapping under the requested device guard is the capability
    // check; do not require the cudaHostAllocMapped bit on UVA systems.
    void* device_pointer = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&device_pointer, cpu.data_ptr(), 0));
    TORCH_CHECK(device_pointer != nullptr, "host allocation has no device mapping");
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(current_device == device_index, "requested mapping device guard was lost");
    return {device_pointer, attr.device, flags, current_device};
}

at::Tensor pinned_alias(const at::Tensor& cpu, int64_t device_index) {
    const auto mapping = checked_host_mapping(cpu, device_index);
    const at::Device target(at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
    // The alias retains its owning CPU Tensor. The explicit owner must also keep
    // it alive until all native kernels and captured graph replays are drained.
    // Official TensorMaker target_device avoids getDeviceFromPtr's allocation-
    // device inference for this validated host mapping. Options and target must
    // agree. Both checks above remain mandatory before this special path.
    return at::for_blob(mapping.device_pointer, cpu.sizes())
        .deleter([keep = cpu](void*) mutable { keep = at::Tensor(); })
        .options(cpu.options().device(target).pinned_memory(false))
        .target_device(target)
        .make_tensor();
}

std::map<std::string, int64_t> pinned_alias_info(const at::Tensor& cpu, int64_t device_index) {
    const auto mapping = checked_host_mapping(cpu, device_index);
    return {{"allocation_device", mapping.allocation_device},
            {"requested_device", device_index}, {"current_device", mapping.current_device},
            {"host_flags", mapping.flags}, {"host_type_verified", 1},
            {"host_pointer", static_cast<int64_t>(reinterpret_cast<uintptr_t>(cpu.data_ptr()))},
            {"mapped_pointer", static_cast<int64_t>(reinterpret_cast<uintptr_t>(mapping.device_pointer))}};
}

static void device_tensor(const at::Tensor& t, at::ScalarType dtype, int device) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.is_contiguous(),
                "expected a contiguous CUDA tensor with the declared dtype");
    TORCH_CHECK(t.get_device() == device, "mixed device tensors are unsupported");
}

void selected_host_q8(
    const at::Tensor& host_q, const at::Tensor& host_s,
    const at::Tensor& source_table, const at::Tensor& slot_generations,
    const at::Tensor& expected_generations, const at::Tensor& row_limits,
    const at::Tensor& indices, const at::Tensor& hot_q, const at::Tensor& hot_s,
    const at::Tensor& remapped, const at::Tensor& hash_keys,
    const at::Tensor& hash_values, const at::Tensor& error, const at::Tensor& metrics) {
    TORCH_CHECK(indices.is_cuda(), "selection must remain on the GPU");
    int device = indices.get_device();
    for (const auto& t : {host_q, source_table, row_limits, indices, hot_q,
                          remapped, hash_keys, hash_values, error, metrics})
        device_tensor(t, at::kInt, device);
    for (const auto& t : {host_s, hot_s}) device_tensor(t, at::kHalf, device);
    for (const auto& t : {slot_generations, expected_generations})
        device_tensor(t, at::kLong, device);
    TORCH_CHECK(host_q.dim() == 3 && host_q.size(1) == 256 && host_q.size(2) == 128,
                "host Q8 geometry must be [pages,256,128] int32");
    TORCH_CHECK(host_s.dim() == 3 && host_s.size(0) == host_q.size(0) &&
                host_s.size(1) == 256 && host_s.size(2) == 16,
                "host scales must be [pages,256,16] FP16");
    TORCH_CHECK(source_table.dim() == 2 && source_table.size(0) == 1 &&
                source_table.size(1) > 0, "one logical source table required");
    TORCH_CHECK(slot_generations.dim() == 1 && slot_generations.numel() == host_q.size(0) &&
                expected_generations.dim() == 1 && expected_generations.numel() == source_table.size(1),
                "generation metadata must cover physical and logical pages");
    TORCH_CHECK(indices.dim() == 2 && 1 <= indices.size(0) && indices.size(0) <= 8 &&
                0 < indices.size(1) && indices.size(1) <= 2080 && indices.size(1) % 32 == 0,
                "decode-only selections must have R1..8 and K<=2080 aligned32");
    TORCH_CHECK(row_limits.dim() == 1 && row_limits.numel() == indices.size(0),
                "one causal visible bound per query row required");
    TORCH_CHECK(hot_q.dim() == 3 && hot_q.size(1) == 256 && hot_q.size(2) == 128 &&
                hot_s.dim() == 3 && hot_s.size(0) == hot_q.size(0) &&
                hot_s.size(1) == 256 && hot_s.size(2) == 16,
                "hot pool must preserve Q8 packed geometry");
    TORCH_CHECK(hot_q.size(0) * 256 >= indices.numel() &&
                remapped.sizes() == indices.sizes(), "fixed hot/remap capacity exceeded");
    TORCH_CHECK(hash_keys.dim() == 1 && hash_keys.numel() >= 2 * indices.numel() &&
                (hash_keys.numel() & (hash_keys.numel() - 1)) == 0 &&
                hash_values.sizes() == hash_keys.sizes(), "hash table must be a bounded power of two");
    TORCH_CHECK(error.numel() == 1 && metrics.numel() == 3,
                "error flag and [unique,valid,probes] metrics required");
    selected_host_q8_cuda(host_q, host_s, source_table, slot_generations,
                          expected_generations, row_limits, indices, hot_q, hot_s,
                          remapped, hash_keys, hash_values, error, metrics);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pinned_alias", &pinned_alias);
    m.def("pinned_alias_info", &pinned_alias_info);
    m.def("selected_host_q8", &selected_host_q8);
}
