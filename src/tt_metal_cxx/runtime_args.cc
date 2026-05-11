#include "tt_metal_cxx/program.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt_stl/span.hpp>

namespace tt_metal_cxx {

namespace {

tt::tt_metal::CoreCoord make_core_coord(std::uint32_t x, std::uint32_t y) {
    return tt::tt_metal::CoreCoord{x, y};
}

std::vector<std::uint32_t> copy_slice(rust::Slice<const std::uint32_t> values) {
    return std::vector<std::uint32_t>(values.begin(), values.end());
}

rust::Vec<std::uint32_t> copy_runtime_args_data(const tt::tt_metal::RuntimeArgsData& runtime_args) {
    rust::Vec<std::uint32_t> result;
    result.reserve(runtime_args.size());
    for (std::size_t index = 0; index < runtime_args.size(); index++) {
        result.push_back(runtime_args[index]);
    }
    return result;
}

}  // namespace

void ProgramHandle::set_runtime_args(
    std::uint32_t kernel_id,
    std::uint32_t core_x,
    std::uint32_t core_y,
    rust::Slice<const std::uint32_t> runtime_args) {
    const auto owned_args = copy_slice(runtime_args);
    const auto core = make_core_coord(core_x, core_y);
    tt::tt_metal::SetRuntimeArgs(
        *program_, kernel_id, core, ttsl::make_const_span(owned_args));
}

rust::Vec<std::uint32_t> ProgramHandle::get_runtime_args(
    std::uint32_t kernel_id, std::uint32_t core_x, std::uint32_t core_y) const {
    const auto core = make_core_coord(core_x, core_y);
    const auto& runtime_args = tt::tt_metal::GetRuntimeArgs(*program_, kernel_id, core);
    return copy_runtime_args_data(runtime_args);
}

void ProgramHandle::set_common_runtime_args(
    std::uint32_t kernel_id, rust::Slice<const std::uint32_t> runtime_args) {
    const auto owned_args = copy_slice(runtime_args);
    tt::tt_metal::SetCommonRuntimeArgs(*program_, kernel_id, ttsl::make_const_span(owned_args));
}

rust::Vec<std::uint32_t> ProgramHandle::get_common_runtime_args(std::uint32_t kernel_id) const {
    const auto& runtime_args = tt::tt_metal::GetCommonRuntimeArgs(*program_, kernel_id);
    return copy_runtime_args_data(runtime_args);
}

}  // namespace tt_metal_cxx
