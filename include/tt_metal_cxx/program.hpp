#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace tt::tt_metal {
class Program;
}

namespace tt_metal_cxx {

class ComputeKernelConfigHandle;
class DataMovementKernelConfigHandle;
class MeshWorkloadHandle;

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

private:
    std::unique_ptr<tt::tt_metal::Program> program_;

    friend class MeshWorkloadHandle;
};

std::unique_ptr<ProgramHandle> create_program();

}  // namespace tt_metal_cxx
