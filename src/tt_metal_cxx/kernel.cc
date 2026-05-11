#include "tt_metal_cxx/kernel.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt_metal_cxx {

ComputeKernelConfigHandle::ComputeKernelConfigHandle() :
    config_(std::make_unique<tt::tt_metal::ComputeConfig>()) {}

ComputeKernelConfigHandle::~ComputeKernelConfigHandle() = default;

void ComputeKernelConfigHandle::set_math_fidelity(std::uint8_t math_fidelity) {
    switch (detail::validate_math_fidelity(math_fidelity)) {
        case 0: config_->math_fidelity = tt::tt_metal::MathFidelity::LoFi; break;
        case 2: config_->math_fidelity = tt::tt_metal::MathFidelity::HiFi2; break;
        case 3: config_->math_fidelity = tt::tt_metal::MathFidelity::HiFi3; break;
        case 4: config_->math_fidelity = tt::tt_metal::MathFidelity::HiFi4; break;
        default: throw std::invalid_argument("unreachable math fidelity");
    }
}

void ComputeKernelConfigHandle::set_fp32_dest_acc_en(bool enabled) {
    config_->fp32_dest_acc_en = enabled;
}

void ComputeKernelConfigHandle::set_dst_full_sync_en(bool enabled) {
    config_->dst_full_sync_en = enabled;
}

void ComputeKernelConfigHandle::fill_unpack_to_dest_modes(std::uint8_t mode) {
    tt::tt_metal::UnpackToDestMode unpack_mode;
    switch (detail::validate_unpack_to_dest_mode(mode)) {
        case 0: unpack_mode = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32; break;
        case 1: unpack_mode = tt::tt_metal::UnpackToDestMode::Default; break;
        default: throw std::invalid_argument("unreachable unpack-to-dest mode");
    }
    config_->unpack_to_dest_mode.assign(64, unpack_mode);
}

void ComputeKernelConfigHandle::set_bfp8_pack_precise(bool enabled) {
    config_->bfp8_pack_precise = enabled;
}

void ComputeKernelConfigHandle::set_math_approx_mode(bool enabled) {
    config_->math_approx_mode = enabled;
}

void ComputeKernelConfigHandle::add_compile_arg(std::uint32_t arg) {
    config_->compile_args.push_back(arg);
}

void ComputeKernelConfigHandle::add_define(rust::Str key, rust::Str value) {
    config_->defines[std::string(key)] = std::string(value);
}

void ComputeKernelConfigHandle::add_named_compile_arg(rust::Str key, std::uint32_t value) {
    config_->named_compile_args[std::string(key)] = value;
}

void ComputeKernelConfigHandle::set_opt_level(std::uint8_t opt_level) {
    switch (detail::validate_kernel_build_opt_level(opt_level)) {
        case 0: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O1; break;
        case 1: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O2; break;
        case 2: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O3; break;
        case 3: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O0; break;
        case 4: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Os; break;
        case 5: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Ofast; break;
        case 6: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Oz; break;
        default: throw std::invalid_argument("unreachable kernel optimization level");
    }
}

DataMovementKernelConfigHandle::DataMovementKernelConfigHandle() :
    config_(std::make_unique<tt::tt_metal::DataMovementConfig>()) {}

DataMovementKernelConfigHandle::DataMovementKernelConfigHandle(tt::tt_metal::DataMovementConfig&& config) :
    config_(std::make_unique<tt::tt_metal::DataMovementConfig>(std::move(config))) {}

DataMovementKernelConfigHandle::~DataMovementKernelConfigHandle() = default;

void DataMovementKernelConfigHandle::set_processor(std::uint8_t processor) {
    switch (detail::validate_data_movement_processor(processor)) {
        case 0: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_0; break;
        case 1: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_1; break;
        case 2: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_2; break;
        case 3: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_3; break;
        case 4: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_4; break;
        case 5: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_5; break;
        case 6: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_6; break;
        case 7: config_->processor = tt::tt_metal::DataMovementProcessor::RISCV_7; break;
        default: throw std::invalid_argument("unreachable data movement processor");
    }
}

void DataMovementKernelConfigHandle::set_noc(std::uint8_t noc) {
    switch (detail::validate_noc(noc)) {
        case 0: config_->noc = tt::tt_metal::NOC::RISCV_0_default; break;
        case 1: config_->noc = tt::tt_metal::NOC::RISCV_1_default; break;
        case 2: config_->noc = tt::tt_metal::NOC::NOC_0; break;
        case 3: config_->noc = tt::tt_metal::NOC::NOC_1; break;
        default: throw std::invalid_argument("unreachable noc selector");
    }
}

void DataMovementKernelConfigHandle::set_noc_mode(std::uint8_t noc_mode) {
    switch (detail::validate_noc_mode(noc_mode)) {
        case 0: config_->noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC; break;
        case 1: config_->noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC; break;
        default: throw std::invalid_argument("unreachable noc mode");
    }
}

void DataMovementKernelConfigHandle::add_compile_arg(std::uint32_t arg) {
    config_->compile_args.push_back(arg);
}

void DataMovementKernelConfigHandle::add_define(rust::Str key, rust::Str value) {
    config_->defines[std::string(key)] = std::string(value);
}

void DataMovementKernelConfigHandle::add_named_compile_arg(rust::Str key, std::uint32_t value) {
    config_->named_compile_args[std::string(key)] = value;
}

void DataMovementKernelConfigHandle::set_opt_level(std::uint8_t opt_level) {
    switch (detail::validate_kernel_build_opt_level(opt_level)) {
        case 0: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O1; break;
        case 1: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O2; break;
        case 2: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O3; break;
        case 3: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::O0; break;
        case 4: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Os; break;
        case 5: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Ofast; break;
        case 6: config_->opt_level = tt::tt_metal::KernelBuildOptLevel::Oz; break;
        default: throw std::invalid_argument("unreachable kernel optimization level");
    }
}

std::unique_ptr<ComputeKernelConfigHandle> create_compute_kernel_config() {
    return std::make_unique<ComputeKernelConfigHandle>();
}

std::unique_ptr<DataMovementKernelConfigHandle> create_data_movement_kernel_config() {
    return std::make_unique<DataMovementKernelConfigHandle>();
}

std::unique_ptr<DataMovementKernelConfigHandle> create_reader_data_movement_kernel_config() {
    return std::make_unique<DataMovementKernelConfigHandle>(tt::tt_metal::ReaderDataMovementConfig{});
}

std::unique_ptr<DataMovementKernelConfigHandle> create_writer_data_movement_kernel_config() {
    return std::make_unique<DataMovementKernelConfigHandle>(tt::tt_metal::WriterDataMovementConfig{});
}

}  // namespace tt_metal_cxx

namespace tt_metal_cxx::detail {

std::uint8_t validate_data_movement_processor(std::uint8_t processor) {
    if (processor > 7) {
        throw std::invalid_argument("data movement processor must be in the range 0..=7");
    }
    return processor;
}

std::uint8_t validate_noc(std::uint8_t noc) {
    if (noc > 3) {
        throw std::invalid_argument("noc selector must be in the range 0..=3");
    }
    return noc;
}

std::uint8_t validate_noc_mode(std::uint8_t noc_mode) {
    if (noc_mode > 1) {
        throw std::invalid_argument("noc mode must be in the range 0..=1");
    }
    return noc_mode;
}

std::uint8_t validate_math_fidelity(std::uint8_t math_fidelity) {
    switch (math_fidelity) {
        case 0:
        case 2:
        case 3:
        case 4: return math_fidelity;
        default: throw std::invalid_argument("math fidelity must be one of 0, 2, 3, or 4");
    }
}

std::uint8_t validate_unpack_to_dest_mode(std::uint8_t unpack_to_dest_mode) {
    if (unpack_to_dest_mode > 1) {
        throw std::invalid_argument("unpack-to-dest mode must be in the range 0..=1");
    }
    return unpack_to_dest_mode;
}

std::uint8_t validate_kernel_build_opt_level(std::uint8_t opt_level) {
    if (opt_level > 6) {
        throw std::invalid_argument("kernel optimization level must be in the range 0..=6");
    }
    return opt_level;
}

}  // namespace tt_metal_cxx::detail
