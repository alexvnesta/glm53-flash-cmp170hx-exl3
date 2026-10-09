"""Owned pinned-request accounting, distinct from allocator/process RAM.

Request power-of-two byte extents explicitly. The tensor views may use less.
This removes the hidden default allocator rounding from the owned request
budget. Cached free blocks, native HostPool and process RSS remain separate.
Importing this module does not import Torch or initialize CUDA.
"""
def pin_request_bytes(payload_bytes):
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise ValueError('positive integer pinned payload required')
    return 1 << (payload_bytes - 1).bit_length()


def allocate_pinned_view(torch, shape, *, dtype, element_size):
    count = 1
    for size in shape:
        if type(size) is not int or size <= 0:
            raise ValueError('positive pinned tensor dimensions required')
        count *= size
    request = pin_request_bytes(count * element_size)
    if request % element_size:
        raise ValueError('pinned request must align to element size')
    extent = torch.empty((request // element_size,), dtype=dtype,
                         device='cpu', pin_memory=True)
    return extent[:count].view(*shape), request
