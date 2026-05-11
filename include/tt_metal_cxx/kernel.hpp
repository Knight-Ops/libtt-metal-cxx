#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace tt::tt_metal {
struct ComputeConfig;
struct DataMovementConfig;
}

namespace tt_metal_cxx {

class ComputeKernelConfigHandle {
public:
    ComputeKernelConfigHandle();
    ~ComputeKernelConfigHandle();

    ComputeKernelConfigHandle(const ComputeKernelConfigHandle&) = delete;
    ComputeKernelConfigHandle& operator=(const ComputeKernelConfigHandle&) = delete;
    ComputeKernelConfigHandle(ComputeKernelConfigHandle&&) = delete;
    ComputeKernelConfigHandle& operator=(ComputeKernelConfigHandle&&) = delete;

    void set_math_fidelity(std::uint8_t math_fidelity);
    void set_fp32_dest_acc_en(bool enabled);
    void set_dst_full_sync_en(bool enabled);
    void fill_unpack_to_dest_modes(std::uint8_t mode);
    void set_bfp8_pack_precise(bool enabled);
    void set_math_approx_mode(bool enabled);
    void add_compile_arg(std::uint32_t arg);
    void add_define(rust::Str key, rust::Str value);
    void add_named_compile_arg(rust::Str key, std::uint32_t value);
    void set_opt_level(std::uint8_t opt_level);

private:
    std::unique_ptr<tt::tt_metal::ComputeConfig> config_;

    friend class ProgramHandle;
};

class DataMovementKernelConfigHandle {
public:
    DataMovementKernelConfigHandle();
    explicit DataMovementKernelConfigHandle(tt::tt_metal::DataMovementConfig&& config);
    ~DataMovementKernelConfigHandle();

    DataMovementKernelConfigHandle(const DataMovementKernelConfigHandle&) = delete;
    DataMovementKernelConfigHandle& operator=(const DataMovementKernelConfigHandle&) = delete;
    DataMovementKernelConfigHandle(DataMovementKernelConfigHandle&&) = delete;
    DataMovementKernelConfigHandle& operator=(DataMovementKernelConfigHandle&&) = delete;

    void set_processor(std::uint8_t processor);
    void set_noc(std::uint8_t noc);
    void set_noc_mode(std::uint8_t noc_mode);
    void add_compile_arg(std::uint32_t arg);
    void add_define(rust::Str key, rust::Str value);
    void add_named_compile_arg(rust::Str key, std::uint32_t value);
    void set_opt_level(std::uint8_t opt_level);

private:
    std::unique_ptr<tt::tt_metal::DataMovementConfig> config_;

    friend class ProgramHandle;
};

std::unique_ptr<ComputeKernelConfigHandle> create_compute_kernel_config();
std::unique_ptr<DataMovementKernelConfigHandle> create_data_movement_kernel_config();
std::unique_ptr<DataMovementKernelConfigHandle> create_reader_data_movement_kernel_config();
std::unique_ptr<DataMovementKernelConfigHandle> create_writer_data_movement_kernel_config();

}  // namespace tt_metal_cxx

namespace tt_metal_cxx::detail {

std::uint8_t validate_data_movement_processor(std::uint8_t processor);
std::uint8_t validate_noc(std::uint8_t noc);
std::uint8_t validate_noc_mode(std::uint8_t noc_mode);
std::uint8_t validate_math_fidelity(std::uint8_t math_fidelity);
std::uint8_t validate_unpack_to_dest_mode(std::uint8_t unpack_to_dest_mode);
std::uint8_t validate_kernel_build_opt_level(std::uint8_t opt_level);

}  // namespace tt_metal_cxx::detail
