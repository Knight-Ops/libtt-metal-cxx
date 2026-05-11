#include "tt_metal_cxx/buffer.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "libtt-metal-cxx/src/ffi.rs.h"
#include "tt_metal_cxx/device.hpp"
#include "tt_metal_cxx/program.hpp"
#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx::detail {

struct BufferBacking {
    BufferBacking(std::shared_ptr<DeviceContext> retained_context, std::shared_ptr<tt::tt_metal::Buffer> retained_buffer) :
        device_context(std::move(retained_context)), buffer(std::move(retained_buffer)) {}

    std::shared_ptr<DeviceContext> device_context;
    std::shared_ptr<tt::tt_metal::Buffer> buffer;
};

}  // namespace tt_metal_cxx::detail

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

tt::tt_metal::TensorMemoryLayout require_tensor_memory_layout(std::uint8_t value) {
    switch (value) {
        case 0: return tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
        case 2: return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
        case 3: return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        case 4: return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        case 5: return tt::tt_metal::TensorMemoryLayout::ND_SHARDED;
        default: throw std::invalid_argument("unsupported TensorMemoryLayout value");
    }
}

tt::tt_metal::ShardOrientation require_shard_orientation(std::uint8_t value) {
    switch (value) {
        case 0: return tt::tt_metal::ShardOrientation::ROW_MAJOR;
        case 1: return tt::tt_metal::ShardOrientation::COL_MAJOR;
        default: throw std::invalid_argument("unsupported ShardOrientation value");
    }
}

tt::DataFormat require_data_format(std::uint8_t value) {
    switch (value) {
        case 0: return tt::DataFormat::Float32;
        case 1: return tt::DataFormat::Float16;
        case 2: return tt::DataFormat::Bfp8;
        case 3: return tt::DataFormat::Bfp4;
        case 4: return tt::DataFormat::Tf32;
        case 5: return tt::DataFormat::Float16_b;
        case 6: return tt::DataFormat::Bfp8_b;
        case 7: return tt::DataFormat::Bfp4_b;
        case 8: return tt::DataFormat::Int32;
        case 9: return tt::DataFormat::UInt16;
        case 10: return tt::DataFormat::Lf8;
        case 11: return tt::DataFormat::Bfp2;
        case 13: return tt::DataFormat::Int16;
        case 14: return tt::DataFormat::Int8;
        case 15: return tt::DataFormat::Bfp2_b;
        case 24: return tt::DataFormat::UInt32;
        case 26: return tt::DataFormat::Fp8_e4m3;
        case 30: return tt::DataFormat::UInt8;
        case 240: return tt::DataFormat::RawUInt8;
        case 241: return tt::DataFormat::RawUInt16;
        case 242: return tt::DataFormat::RawUInt32;
        case 255: return tt::DataFormat::Invalid;
        default: throw std::invalid_argument("unsupported DataFormat value");
    }
}

tt::tt_metal::Tile make_tile(std::uint32_t height, std::uint32_t width, bool transpose_tile) {
    return tt::tt_metal::Tile({height, width}, transpose_tile);
}

std::shared_ptr<detail::DeviceContext> require_device_context(const DeviceHandle& device) {
    auto context = device.context();
    if (context == nullptr || device.raw_device() == nullptr) {
        throw std::invalid_argument("device must be open to create TT-Metal buffers");
    }
    return context;
}

const std::shared_ptr<detail::BufferBacking>& require_buffer_backing(const BufferHandle& buffer) {
    if (buffer.backing() == nullptr || buffer.backing()->buffer == nullptr) {
        throw std::invalid_argument("buffer handle is invalid");
    }
    return buffer.backing();
}

tt::tt_metal::Buffer& require_buffer(BufferHandle& buffer) {
    return *require_buffer_backing(buffer)->buffer;
}

const tt::tt_metal::Buffer& require_buffer(const BufferHandle& buffer) {
    return *require_buffer_backing(buffer)->buffer;
}

std::vector<tt::tt_metal::CoreRange> make_core_ranges(rust::Slice<const CoreRangeRepr> ranges) {
    std::vector<tt::tt_metal::CoreRange> core_ranges;
    core_ranges.reserve(ranges.size());
    for (const auto& range : ranges) {
        core_ranges.emplace_back(
            tt::tt_metal::CoreCoord{range.start_x, range.start_y},
            tt::tt_metal::CoreCoord{range.end_x, range.end_y});
    }
    return core_ranges;
}

tt::tt_metal::CoreRangeSet make_core_range_set(rust::Slice<const CoreRangeRepr> ranges) {
    return tt::tt_metal::CoreRangeSet(make_core_ranges(ranges));
}

tt::tt_metal::ShardSpecBuffer make_shard_spec_buffer(
    const ShardedBufferConfigRepr& config, rust::Slice<const CoreRangeRepr> core_ranges) {
    return tt::tt_metal::ShardSpecBuffer(
        make_core_range_set(core_ranges),
        {config.shard_shape_height, config.shard_shape_width},
        require_shard_orientation(config.shard_orientation),
        {config.page_shape_height, config.page_shape_width},
        {config.tensor2d_shape_height, config.tensor2d_shape_width});
}

tt::tt_metal::SubDeviceId make_sub_device_id(std::uint8_t value) {
    return tt::tt_metal::SubDeviceId(value);
}

std::shared_ptr<tt::tt_metal::Buffer> create_buffer_with_options(
    const tt::tt_metal::InterleavedBufferConfig& config, const BufferCreateOptionsRepr& options) {
    if (options.has_address && options.has_sub_device_id) {
        throw std::invalid_argument("buffer create options cannot contain both address and sub-device id");
    }

    if (options.has_address) {
        return tt::tt_metal::CreateBuffer(config, options.address);
    }
    if (options.has_sub_device_id) {
        return tt::tt_metal::CreateBuffer(config, make_sub_device_id(options.sub_device_id));
    }
    return tt::tt_metal::CreateBuffer(config);
}

std::shared_ptr<tt::tt_metal::Buffer> create_buffer_with_options(
    const tt::tt_metal::ShardedBufferConfig& config, const BufferCreateOptionsRepr& options) {
    if (options.has_address && options.has_sub_device_id) {
        throw std::invalid_argument("buffer create options cannot contain both address and sub-device id");
    }

    if (options.has_address) {
        return tt::tt_metal::CreateBuffer(config, options.address);
    }
    if (options.has_sub_device_id) {
        return tt::tt_metal::CreateBuffer(config, make_sub_device_id(options.sub_device_id));
    }
    return tt::tt_metal::CreateBuffer(config);
}

tt::tt_metal::CircularBufferConfig::Builder make_cb_builder(
    tt::tt_metal::CircularBufferConfig& config, std::uint8_t buffer_index, bool remote) {
    return remote ? config.remote_index(buffer_index) : config.index(buffer_index);
}

bool tile_is_transposed(const tt::tt_metal::Tile& tile) {
    return tile.get_transpose_within_face() && tile.get_transpose_of_faces();
}

}  // namespace

BufferHandle::BufferHandle(std::shared_ptr<detail::BufferBacking> backing) noexcept : backing_(std::move(backing)) {}

BufferInfoRepr BufferHandle::info() const {
    const auto& buffer = require_buffer(*this);
    const auto sub_device_id = buffer.sub_device_id();
    return BufferInfoRepr{
        .is_allocated = buffer.is_allocated(),
        .address = buffer.address(),
        .size = buffer.size(),
        .page_size = buffer.page_size(),
        .buffer_type = static_cast<std::uint8_t>(buffer.buffer_type()),
        .buffer_layout = static_cast<std::uint8_t>(buffer.buffer_layout()),
        .has_sub_device_id = sub_device_id.has_value(),
        .sub_device_id = static_cast<std::uint8_t>(sub_device_id.has_value() ? *(*sub_device_id) : 0),
    };
}

bool BufferHandle::deallocate() {
    auto& buffer = require_buffer(*this);
    if (!buffer.is_allocated()) {
        return false;
    }
    tt::tt_metal::DeallocateBuffer(buffer);
    return true;
}

bool BufferHandle::has_shard_spec() const {
    return require_buffer(*this).has_shard_spec();
}

ShardSpecBufferMetadataRepr BufferHandle::shard_spec_metadata() const {
    const auto& buffer = require_buffer(*this);
    if (!buffer.has_shard_spec()) {
        throw std::invalid_argument("buffer does not have a shard spec");
    }
    const auto shard_spec = buffer.shard_spec();
    return ShardSpecBufferMetadataRepr{
        .shard_shape_height = shard_spec.shape()[0],
        .shard_shape_width = shard_spec.shape()[1],
        .shard_orientation = static_cast<std::uint8_t>(shard_spec.orientation()),
        .page_shape_height = shard_spec.page_shape[0],
        .page_shape_width = shard_spec.page_shape[1],
        .tensor2d_shape_height = shard_spec.tensor2d_shape_in_pages[0],
        .tensor2d_shape_width = shard_spec.tensor2d_shape_in_pages[1],
    };
}

rust::Vec<CoreRangeRepr> BufferHandle::shard_spec_core_ranges() const {
    const auto& buffer = require_buffer(*this);
    if (!buffer.has_shard_spec()) {
        throw std::invalid_argument("buffer does not have a shard spec");
    }

    const auto shard_spec = buffer.shard_spec();
    const auto grid = shard_spec.grid();
    rust::Vec<CoreRangeRepr> ranges;
    for (const auto& range : grid.ranges()) {
        ranges.push_back(CoreRangeRepr{
            .start_x = static_cast<std::uint32_t>(range.start_coord.x),
            .start_y = static_cast<std::uint32_t>(range.start_coord.y),
            .end_x = static_cast<std::uint32_t>(range.end_coord.x),
            .end_y = static_cast<std::uint32_t>(range.end_coord.y),
        });
    }
    return ranges;
}

const std::shared_ptr<detail::BufferBacking>& BufferHandle::backing() const noexcept {
    return backing_;
}

CircularBufferConfigHandle::CircularBufferConfigHandle(std::uint32_t total_size) :
    config_(std::make_unique<tt::tt_metal::CircularBufferConfig>(total_size)) {}

CircularBufferConfigHandle::~CircularBufferConfigHandle() = default;

void CircularBufferConfigHandle::set_total_size(std::uint32_t total_size) {
    config_->set_total_size(total_size);
}

void CircularBufferConfigHandle::set_address_offset(std::uint32_t offset) {
    config_->set_address_offset(offset);
}

void CircularBufferConfigHandle::set_globally_allocated_address(const BufferHandle& buffer) {
    global_buffer_ = require_buffer_backing(buffer);
    config_->set_globally_allocated_address(*global_buffer_->buffer);
}

void CircularBufferConfigHandle::set_globally_allocated_address_and_total_size(
    const BufferHandle& buffer, std::uint32_t total_size) {
    global_buffer_ = require_buffer_backing(buffer);
    config_->set_globally_allocated_address_and_total_size(*global_buffer_->buffer, total_size);
}

void CircularBufferConfigHandle::set_index_data_format(
    std::uint8_t buffer_index, bool remote, std::uint8_t data_format) {
    make_cb_builder(*config_, buffer_index, remote).set_data_format(require_data_format(data_format));
}

void CircularBufferConfigHandle::set_index_total_size(
    std::uint8_t buffer_index, bool remote, std::uint32_t total_size) {
    make_cb_builder(*config_, buffer_index, remote).set_total_size(total_size);
}

void CircularBufferConfigHandle::set_index_page_size(
    std::uint8_t buffer_index, bool remote, std::uint32_t page_size) {
    make_cb_builder(*config_, buffer_index, remote).set_page_size(page_size);
}

void CircularBufferConfigHandle::set_index_tile(
    std::uint8_t buffer_index,
    bool remote,
    std::uint32_t tile_height,
    std::uint32_t tile_width,
    bool transpose_tile) {
    make_cb_builder(*config_, buffer_index, remote).set_tile_dims(make_tile(tile_height, tile_width, transpose_tile));
}

std::unique_ptr<BufferHandle> create_interleaved_buffer(
    const DeviceHandle& device, const InterleavedBufferConfigRepr& config, const BufferCreateOptionsRepr& options) {
    detail::ensure_runtime_root();
    detail::register_atexit_cleanup();

    auto context = require_device_context(device);
    tt::tt_metal::InterleavedBufferConfig native_config{
        .device = device.raw_device(),
        .size = config.size,
        .page_size = config.page_size,
        .buffer_type = require_buffer_type(config.buffer_type),
    };

    return std::make_unique<BufferHandle>(std::make_shared<detail::BufferBacking>(
        std::move(context), create_buffer_with_options(native_config, options)));
}

std::unique_ptr<BufferHandle> create_sharded_buffer(
    const DeviceHandle& device,
    const ShardedBufferConfigRepr& config,
    rust::Slice<const CoreRangeRepr> core_ranges,
    const BufferCreateOptionsRepr& options) {
    detail::ensure_runtime_root();
    detail::register_atexit_cleanup();

    auto context = require_device_context(device);
    tt::tt_metal::ShardedBufferConfig native_config{
        .device = device.raw_device(),
        .size = config.size,
        .page_size = config.page_size,
        .buffer_type = require_buffer_type(config.buffer_type),
        .buffer_layout = require_tensor_memory_layout(config.buffer_layout),
        .shard_parameters = make_shard_spec_buffer(config, core_ranges),
    };

    return std::make_unique<BufferHandle>(std::make_shared<detail::BufferBacking>(
        std::move(context), create_buffer_with_options(native_config, options)));
}

std::unique_ptr<CircularBufferConfigHandle> create_circular_buffer_config(std::uint32_t total_size) {
    return std::make_unique<CircularBufferConfigHandle>(total_size);
}

void ProgramHandle::retain_buffer(const std::shared_ptr<detail::BufferBacking>& buffer) {
    if (buffer != nullptr) {
        retained_buffers_.push_back(buffer);
    }
}

void ProgramHandle::assign_global_buffer(const BufferHandle& buffer) {
    const auto& backing = require_buffer_backing(buffer);
    tt::tt_metal::AssignGlobalBufferToProgram(backing->buffer, *program_);
    retain_buffer(backing);
}

std::uintptr_t ProgramHandle::create_circular_buffer(
    rust::Slice<const CoreRangeRepr> core_ranges, const CircularBufferConfigHandle& config) {
    if (config.config_ == nullptr) {
        throw std::invalid_argument("circular buffer config is invalid");
    }

    const auto handle = tt::tt_metal::CreateCircularBuffer(*program_, make_core_range_set(core_ranges), *config.config_);
    retain_buffer(config.global_buffer_);
    return handle;
}

CircularBufferMetadataRepr ProgramHandle::get_circular_buffer_metadata(std::uintptr_t cb_handle) const {
    const auto& config = tt::tt_metal::GetCircularBufferConfig(*program_, cb_handle);
    const auto address = config.globally_allocated_address();
    return CircularBufferMetadataRepr{
        .total_size = config.total_size(),
        .has_globally_allocated_address = address.has_value(),
        .globally_allocated_address = address.value_or(0),
        .dynamic_cb = config.dynamic_cb(),
        .max_size = config.max_size(),
        .buffer_size = config.buffer_size(),
        .address_offset = config.address_offset(),
    };
}

rust::Vec<CircularBufferIndexConfigRepr> ProgramHandle::get_circular_buffer_indices(std::uintptr_t cb_handle) const {
    const auto& config = tt::tt_metal::GetCircularBufferConfig(*program_, cb_handle);
    std::vector<std::uint8_t> indices(config.buffer_indices().begin(), config.buffer_indices().end());
    std::sort(indices.begin(), indices.end());

    rust::Vec<CircularBufferIndexConfigRepr> result;
    for (const auto index : indices) {
        const auto& data_formats = config.data_formats();
        const auto& page_sizes = config.page_sizes();
        const auto& tiles = config.tiles();
        const auto tile = tiles[index];
        result.push_back(CircularBufferIndexConfigRepr{
            .buffer_index = index,
            .is_remote = config.remote_buffer_indices().contains(index),
            .has_data_format = data_formats[index].has_value(),
            .data_format = static_cast<std::uint8_t>(
                data_formats[index].has_value() ? static_cast<std::uint8_t>(*data_formats[index]) : 0),
            .has_page_size = page_sizes[index].has_value(),
            .page_size = page_sizes[index].value_or(0),
            .has_tile = tile.has_value(),
            .tile_height = tile.has_value() ? tile->get_height() : 0,
            .tile_width = tile.has_value() ? tile->get_width() : 0,
            .tile_transpose = tile.has_value() ? tile_is_transposed(*tile) : false,
        });
    }
    return result;
}

void ProgramHandle::update_circular_buffer_total_size(std::uintptr_t cb_handle, std::uint32_t total_size) {
    tt::tt_metal::UpdateCircularBufferTotalSize(*program_, cb_handle, total_size);
}

void ProgramHandle::update_circular_buffer_page_size(
    std::uintptr_t cb_handle, std::uint8_t buffer_index, std::uint32_t page_size) {
    tt::tt_metal::UpdateCircularBufferPageSize(*program_, cb_handle, buffer_index, page_size);
}

void ProgramHandle::update_dynamic_circular_buffer_address(std::uintptr_t cb_handle, const BufferHandle& buffer) {
    const auto& backing = require_buffer_backing(buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(*program_, cb_handle, *backing->buffer);
    retain_buffer(backing);
}

void ProgramHandle::update_dynamic_circular_buffer_address_with_offset(
    std::uintptr_t cb_handle, const BufferHandle& buffer, std::uint32_t address_offset) {
    const auto& backing = require_buffer_backing(buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(*program_, cb_handle, *backing->buffer, address_offset);
    retain_buffer(backing);
}

void ProgramHandle::update_dynamic_circular_buffer_address_and_total_size(
    std::uintptr_t cb_handle, const BufferHandle& buffer, std::uint32_t total_size) {
    const auto& backing = require_buffer_backing(buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddressAndTotalSize(*program_, cb_handle, *backing->buffer, total_size);
    retain_buffer(backing);
}

std::uint32_t ProgramHandle::create_semaphore(
    rust::Slice<const CoreRangeRepr> core_ranges, std::uint32_t initial_value) {
    return tt::tt_metal::CreateSemaphore(*program_, make_core_range_set(core_ranges), initial_value);
}

}  // namespace tt_metal_cxx
