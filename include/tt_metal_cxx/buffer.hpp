#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace tt::tt_metal {
class Buffer;
class CircularBufferConfig;
}

namespace tt_metal_cxx {

struct BufferCreateOptionsRepr;
struct BufferInfoRepr;
struct CoreRangeRepr;
struct InterleavedBufferConfigRepr;
struct ShardSpecBufferMetadataRepr;
struct ShardedBufferConfigRepr;

class DeviceHandle;
class ProgramHandle;

namespace detail {
struct BufferBacking;
}

class BufferHandle {
public:
    explicit BufferHandle(std::shared_ptr<detail::BufferBacking> backing) noexcept;
    ~BufferHandle() = default;

    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    BufferHandle(BufferHandle&&) = delete;
    BufferHandle& operator=(BufferHandle&&) = delete;

    BufferInfoRepr info() const;
    bool deallocate();
    bool has_shard_spec() const;
    ShardSpecBufferMetadataRepr shard_spec_metadata() const;
    rust::Vec<CoreRangeRepr> shard_spec_core_ranges() const;
    const std::shared_ptr<detail::BufferBacking>& backing() const noexcept;

private:
    std::shared_ptr<detail::BufferBacking> backing_;

    friend class CircularBufferConfigHandle;
    friend class ProgramHandle;
};

class CircularBufferConfigHandle {
public:
    explicit CircularBufferConfigHandle(std::uint32_t total_size);
    ~CircularBufferConfigHandle();

    CircularBufferConfigHandle(const CircularBufferConfigHandle&) = delete;
    CircularBufferConfigHandle& operator=(const CircularBufferConfigHandle&) = delete;
    CircularBufferConfigHandle(CircularBufferConfigHandle&&) = delete;
    CircularBufferConfigHandle& operator=(CircularBufferConfigHandle&&) = delete;

    void set_total_size(std::uint32_t total_size);
    void set_address_offset(std::uint32_t offset);
    void set_globally_allocated_address(const BufferHandle& buffer);
    void set_globally_allocated_address_and_total_size(const BufferHandle& buffer, std::uint32_t total_size);
    void set_index_data_format(std::uint8_t buffer_index, bool remote, std::uint8_t data_format);
    void set_index_total_size(std::uint8_t buffer_index, bool remote, std::uint32_t total_size);
    void set_index_page_size(std::uint8_t buffer_index, bool remote, std::uint32_t page_size);
    void set_index_tile(
        std::uint8_t buffer_index,
        bool remote,
        std::uint32_t tile_height,
        std::uint32_t tile_width,
        bool transpose_tile);

private:
    std::shared_ptr<detail::BufferBacking> global_buffer_;
    std::unique_ptr<tt::tt_metal::CircularBufferConfig> config_;

    friend class ProgramHandle;
};

std::unique_ptr<BufferHandle> create_interleaved_buffer(
    const DeviceHandle& device,
    const InterleavedBufferConfigRepr& config,
    const BufferCreateOptionsRepr& options);
std::unique_ptr<BufferHandle> create_sharded_buffer(
    const DeviceHandle& device,
    const ShardedBufferConfigRepr& config,
    rust::Slice<const CoreRangeRepr> core_ranges,
    const BufferCreateOptionsRepr& options);
std::unique_ptr<CircularBufferConfigHandle> create_circular_buffer_config(std::uint32_t total_size);

}  // namespace tt_metal_cxx
