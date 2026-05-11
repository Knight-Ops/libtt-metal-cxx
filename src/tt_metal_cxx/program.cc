#include "tt_metal_cxx/program.hpp"

#include <memory>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "tt_metal_cxx/kernel.hpp"
#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx {

namespace {

tt::tt_metal::CoreCoord make_core_coord(std::uint32_t x, std::uint32_t y) {
    return tt::tt_metal::CoreCoord{x, y};
}

std::string require_kernel_path(rust::Str file_name) {
    if (file_name.empty()) {
        throw std::invalid_argument("kernel file path must not be empty");
    }
    return std::string(file_name);
}

}  // namespace

ProgramHandle::ProgramHandle(tt::tt_metal::Program&& program) noexcept :
    program_(std::make_unique<tt::tt_metal::Program>(std::move(program))) {
    detail::note_program_opened();
}

ProgramHandle::~ProgramHandle() {
    try {
        if (program_ != nullptr) {
            program_.reset();
            detail::note_program_closed();
            detail::maybe_release_ownership();
        }
    } catch (...) {
    }
}

std::uint64_t ProgramHandle::runtime_id() const {
    return program_->get_runtime_id();
}

void ProgramHandle::set_runtime_id(std::uint64_t runtime_id) {
    program_->set_runtime_id(runtime_id);
}

std::uint32_t ProgramHandle::create_compute_kernel(rust::Str file_name, std::uint32_t core_x, std::uint32_t core_y) {
    return create_compute_kernel_with_config(
        file_name, core_x, core_y, *create_compute_kernel_config());
}

std::uint32_t ProgramHandle::create_compute_kernel_with_config(
    rust::Str file_name,
    std::uint32_t core_x,
    std::uint32_t core_y,
    const ComputeKernelConfigHandle& config) {
    if (config.config_ == nullptr) {
        throw std::invalid_argument("compute kernel config is invalid");
    }

    const auto kernel_id = tt::tt_metal::CreateKernel(
        *program_,
        require_kernel_path(file_name),
        make_core_coord(core_x, core_y),
        *config.config_);
    return kernel_id;
}

std::uint32_t ProgramHandle::create_data_movement_kernel(
    rust::Str file_name,
    std::uint32_t core_x,
    std::uint32_t core_y,
    std::uint8_t processor,
    std::uint8_t noc) {
    auto config = create_data_movement_kernel_config();
    config->set_processor(processor);
    config->set_noc(noc);
    return create_data_movement_kernel_with_config(file_name, core_x, core_y, *config);
}

std::uint32_t ProgramHandle::create_data_movement_kernel_with_config(
    rust::Str file_name,
    std::uint32_t core_x,
    std::uint32_t core_y,
    const DataMovementKernelConfigHandle& config) {
    if (config.config_ == nullptr) {
        throw std::invalid_argument("data movement kernel config is invalid");
    }

    const auto kernel_id = tt::tt_metal::CreateKernel(
        *program_,
        require_kernel_path(file_name),
        make_core_coord(core_x, core_y),
        *config.config_);
    return kernel_id;
}

std::unique_ptr<ProgramHandle> create_program() {
    detail::ensure_runtime_root();
    detail::register_atexit_cleanup();
    return std::make_unique<ProgramHandle>(tt::tt_metal::CreateProgram());
}

}  // namespace tt_metal_cxx
