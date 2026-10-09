#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// Error bits:1 invalid/future logical row;2 absent physical page;4 stale page
// generation;8 hash/mapping failure. No input text or CPU selection readback.
__global__ void gather_unique(
    const int32_t* source_q, const __half* source_s, const int32_t* table,
    const int64_t* slot_epochs, const int64_t* logical_epochs,
    const int32_t* limits, const int32_t* indices, int32_t* hot_q, __half* hot_s,
    int32_t* keys, int32_t* values, int32_t* error, int32_t* metrics,
    int width, int logical_pages, int physical_pages, int table_size) {
    int flat = blockIdx.x;
    __shared__ int winner;
    __shared__ int physical_row;
    if (threadIdx.x == 0) {
        winner = 0;
        int row = indices[flat];
        if (row != -1) {
            if (row < 0 || row >= limits[flat / width] || row / 256 >= logical_pages) {
                atomicOr(error, 1);
            } else {
                int logical_page = row / 256;
                int physical_page = table[logical_page];
                if (physical_page < 0 || physical_page >= physical_pages) {
                    atomicOr(error, 2);
                } else if (slot_epochs[physical_page] != logical_epochs[logical_page] ||
                           logical_epochs[logical_page] <= 0) {
                    atomicOr(error, 4);
                } else {
                    atomicAdd(metrics + 1, 1);
                    uint32_t pos = (static_cast<uint32_t>(row) * 2654435761U) & (table_size - 1);
                    bool found = false;
                    for (int probe = 0; probe < table_size; ++probe) {
                        atomicAdd(metrics + 2, 1);
                        int old = atomicCAS(keys + pos, -1, row);
                        if (old == -1) {
                            values[pos] = flat;
                            physical_row = physical_page * 256 + row % 256;
                            winner = 1;
                            atomicAdd(metrics, 1);
                            found = true;
                            break;
                        }
                        if (old == row) { found = true; break; }
                        pos = (pos + 1) & (table_size - 1);
                    }
                    if (!found) atomicOr(error, 8);
                }
            }
        }
    }
    __syncthreads();
    if (winner) {
        // All128 int32 words and16 scale words are bitwise copied once per
        // logical row. Winner flat slots may contain holes; remapping hides
        // them from DSA and preserves duplicates and original selection order.
        int lane = threadIdx.x;
        if (lane < 128) hot_q[flat * 128 + lane] = source_q[physical_row * 128 + lane];
        if (lane < 16) hot_s[flat * 16 + lane] = source_s[physical_row * 16 + lane];
    }
}

__global__ void remap_rows(const int32_t* indices, int32_t* remapped,
    const int32_t* keys, const int32_t* values, const int32_t* error,
    int32_t* write_error, int count, int table_size) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    int row = indices[flat];
    remapped[flat] = -1;
    // Any guarded failure invalidates the whole step. The owner must inspect
    // the error AFTER its completion event before publishing generated output.
    if (row == -1 || *error != 0) return;
    uint32_t pos = (static_cast<uint32_t>(row) * 2654435761U) & (table_size - 1);
    for (int probe = 0; probe < table_size; ++probe) {
        int key = keys[pos];
        if (key == row) { remapped[flat] = values[pos]; return; }
        if (key == -1) break;
        pos = (pos + 1) & (table_size - 1);
    }
    atomicOr(write_error, 8);
}

void selected_host_q8_cuda(
    const at::Tensor& q, const at::Tensor& s, const at::Tensor& table,
    const at::Tensor& slot_epochs, const at::Tensor& logical_epochs,
    const at::Tensor& limits, const at::Tensor& indices,
    const at::Tensor& hot_q, const at::Tensor& hot_s, const at::Tensor& remapped,
    const at::Tensor& keys, const at::Tensor& values,
    const at::Tensor& error, const at::Tensor& metrics) {
    c10::cuda::CUDAGuard guard(indices.device());
    auto stream = at::cuda::getCurrentCUDAStream(indices.get_device());
    int count = static_cast<int>(indices.numel());
    gather_unique<<<count, 128, 0, stream>>>(
        q.data_ptr<int32_t>(), reinterpret_cast<const __half*>(s.data_ptr<at::Half>()),
        table.data_ptr<int32_t>(), slot_epochs.data_ptr<int64_t>(),
        logical_epochs.data_ptr<int64_t>(), limits.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), hot_q.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(hot_s.data_ptr<at::Half>()), keys.data_ptr<int32_t>(),
        values.data_ptr<int32_t>(), error.data_ptr<int32_t>(), metrics.data_ptr<int32_t>(),
        indices.size(1), table.size(1), q.size(0), keys.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    remap_rows<<<(count + 255) / 256, 256, 0, stream>>>(
        indices.data_ptr<int32_t>(), remapped.data_ptr<int32_t>(),
        keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(),
        error.data_ptr<int32_t>(), error.data_ptr<int32_t>(), count, keys.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
