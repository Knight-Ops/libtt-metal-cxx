#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rust/cxx.h"

namespace tt::tt_metal {
class Program;
}

namespace tt_metal_cxx {

class BufferHandle;
class ComputeKernelConfigHandle;
class CircularBufferConfigHandle;
class DataMovementKernelConfigHandle;
class MeshWorkloadHandle;

namespace detail {
struct BufferBacking;
}

struct CircularBufferIndexConfigRepr;
struct CircularBufferMetadataRepr;
struct CoreRangeRepr;

class ProgramHandle {
public:
    explicit ProgramHandle(tt::tt_metal::Program&& program) noexcept;
    ~ProgramHandle();

    ProgramHandle(const ProgramHandle&) = delete;
    ProgramHandle& operator=(const ProgramHandle&) = delete;
    ProgramHandle(ProgramHandle&&) = delete;
    ProgramHandle& operator=(ProgramHandle&&) = delete;

    std::uint64_t runtime_id() const;
    void set_runtime_id(std::uint64_t runtime_id);
    void set_runtime_args(
        std::uint32_t kernel_id,
        std::uint32_t core_x,
        std::uint32_t core_y,
        rust::Slice<const std::uint32_t> runtime_args);
    rust::Vec<std::uint32_t> get_runtime_args(std::uint32_t kernel_id, std::uint32_t core_x, std::uint32_t core_y) const;
    void set_common_runtime_args(std::uint32_t kernel_id, rust::Slice<const std::uint32_t> runtime_args);
    rust::Vec<std::uint32_t> get_common_runtime_args(std::uint32_t kernel_id) const;
    std::uint32_t create_compute_kernel(rust::Str file_name, std::uint32_t core_x, std::uint32_t core_y);
    std::uint32_t create_compute_kernel_from_string(
        rust::Str kernel_src_code, std::uint32_t core_x, std::uint32_t core_y);
    std::uint32_t create_compute_kernel_with_config(
        rust::Str file_name,
        std::uint32_t core_x,
        std::uint32_t core_y,
        const ComputeKernelConfigHandle& config);
    std::uint32_t create_compute_kernel_from_string_with_config(
        rust::Str kernel_src_code,
        std::uint32_t core_x,
        std::uint32_t core_y,
        const ComputeKernelConfigHandle& config);
    std::uint32_t create_data_movement_kernel(
        rust::Str file_name,
        std::uint32_t core_x,
        std::uint32_t core_y,
        std::uint8_t processor,
        std::uint8_t noc);
    std::uint32_t create_data_movement_kernel_from_string(
        rust::Str kernel_src_code,
        std::uint32_t core_x,
        std::uint32_t core_y,
        std::uint8_t processor,
        std::uint8_t noc);
    std::uint32_t create_data_movement_kernel_with_config(
        rust::Str file_name,
        std::uint32_t core_x,
        std::uint32_t core_y,
        const DataMovementKernelConfigHandle& config);
    std::uint32_t create_data_movement_kernel_from_string_with_config(
        rust::Str kernel_src_code,
        std::uint32_t core_x,
        std::uint32_t core_y,
        const DataMovementKernelConfigHandle& config);
    void assign_global_buffer(const BufferHandle& buffer);
    std::uintptr_t create_circular_buffer(
        rust::Slice<const CoreRangeRepr> core_ranges, const CircularBufferConfigHandle& config);
    CircularBufferMetadataRepr get_circular_buffer_metadata(std::uintptr_t cb_handle) const;
    rust::Vec<CircularBufferIndexConfigRepr> get_circular_buffer_indices(std::uintptr_t cb_handle) const;
    void update_circular_buffer_total_size(std::uintptr_t cb_handle, std::uint32_t total_size);
    void update_circular_buffer_page_size(
        std::uintptr_t cb_handle, std::uint8_t buffer_index, std::uint32_t page_size);
    void update_dynamic_circular_buffer_address(std::uintptr_t cb_handle, const BufferHandle& buffer);
    void update_dynamic_circular_buffer_address_with_offset(
        std::uintptr_t cb_handle, const BufferHandle& buffer, std::uint32_t address_offset);
    void update_dynamic_circular_buffer_address_and_total_size(
        std::uintptr_t cb_handle, const BufferHandle& buffer, std::uint32_t total_size);
    std::uint32_t create_semaphore(rust::Slice<const CoreRangeRepr> core_ranges, std::uint32_t initial_value);

private:
    void retain_buffer(const std::shared_ptr<detail::BufferBacking>& buffer);

    std::vector<std::shared_ptr<detail::BufferBacking>> retained_buffers_;
    std::unique_ptr<tt::tt_metal::Program> program_;

    friend class MeshWorkloadHandle;
};

std::unique_ptr<ProgramHandle> create_program();

}  // namespace tt_metal_cxx
