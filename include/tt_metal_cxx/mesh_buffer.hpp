#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace tt::tt_metal::distributed {
class MeshBuffer;
class MeshDevice;
}

namespace tt_metal_cxx {

class MeshDeviceHandle;

class MeshBufferHandle {
public:
    ~MeshBufferHandle();

    MeshBufferHandle(const MeshBufferHandle&) = delete;
    MeshBufferHandle& operator=(const MeshBufferHandle&) = delete;
    MeshBufferHandle(MeshBufferHandle&&) = delete;
    MeshBufferHandle& operator=(MeshBufferHandle&&) = delete;

    std::uint32_t address() const;
    std::uint64_t size() const;
    bool is_allocated() const;

private:
    explicit MeshBufferHandle(
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer) noexcept;

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer_;

    friend class MeshDeviceHandle;
    friend std::unique_ptr<MeshBufferHandle> create_replicated_mesh_buffer(
        const MeshDeviceHandle& mesh_device,
        std::uint64_t size_bytes,
        std::uint64_t page_size,
        std::uint8_t buffer_type);
};

std::unique_ptr<MeshBufferHandle> create_replicated_mesh_buffer(
    const MeshDeviceHandle& mesh_device,
    std::uint64_t size_bytes,
    std::uint64_t page_size,
    std::uint8_t buffer_type);

}  // namespace tt_metal_cxx
