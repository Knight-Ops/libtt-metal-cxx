#include "tt_metal_cxx/mesh_buffer.hpp"

#include <memory>
#include <stdexcept>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tt_metal_cxx/distributed.hpp"
#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx {

namespace {

tt::tt_metal::BufferType require_buffer_type(std::uint8_t value) {
    switch (value) {
        case 0: return tt::tt_metal::BufferType::DRAM;
        case 1: return tt::tt_metal::BufferType::L1;
        case 2: return tt::tt_metal::BufferType::SYSTEM_MEMORY;
        case 3: return tt::tt_metal::BufferType::L1_SMALL;
        case 4: return tt::tt_metal::BufferType::TRACE;
        default: throw std::invalid_argument("unsupported BufferType value");
    }
}

}  // namespace

MeshBufferHandle::MeshBufferHandle(
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer) noexcept :
    buffer_(std::move(buffer)) {}

MeshBufferHandle::~MeshBufferHandle() = default;

std::unique_ptr<MeshBufferHandle> create_replicated_mesh_buffer(
    const MeshDeviceHandle& mesh_device,
    std::uint64_t size_bytes,
    std::uint64_t page_size,
    std::uint8_t buffer_type) {
    detail::ensure_runtime_root();
    detail::register_atexit_cleanup();

    if (!mesh_device.is_open()) {
        throw std::runtime_error("mesh device is closed");
    }

    auto buffer_type_tt = require_buffer_type(buffer_type);

    auto mesh_buffer_config = tt::tt_metal::distributed::ReplicatedBufferConfig{
        .size = static_cast<tt::tt_metal::DeviceAddr>(size_bytes),
    };

    auto device_local_config = tt::tt_metal::distributed::DeviceLocalBufferConfig{
        .page_size = static_cast<tt::tt_metal::DeviceAddr>(page_size),
        .buffer_type = buffer_type_tt,
        .sharding_args = {},
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };

    auto mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        mesh_buffer_config, device_local_config, mesh_device.mesh_device_.get());

    if (mesh_buffer == nullptr) {
        throw std::runtime_error("MeshBuffer::create returned null");
    }

    return std::unique_ptr<MeshBufferHandle>(new MeshBufferHandle(std::move(mesh_buffer)));
}

std::uint32_t MeshBufferHandle::address() const {
    if (buffer_ == nullptr) {
        throw std::runtime_error("mesh buffer handle is invalid");
    }
    return static_cast<std::uint32_t>(buffer_->address());
}

std::uint64_t MeshBufferHandle::size() const {
    if (buffer_ == nullptr) {
        throw std::runtime_error("mesh buffer handle is invalid");
    }
    return buffer_->size();
}

bool MeshBufferHandle::is_allocated() const {
    return buffer_ != nullptr && buffer_->is_allocated();
}

}  // namespace tt_metal_cxx
