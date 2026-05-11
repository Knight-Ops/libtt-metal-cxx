#include "include/tt_metal_cxx.hpp"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string_view>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_metal::detail {
void ReleaseOwnership();
}

namespace tt_metal_cxx {

namespace {

bool is_runtime_root(const std::filesystem::path& candidate) {
    return !candidate.empty() && std::filesystem::is_directory(candidate / "tt_metal");
}

bool set_root_dir_if_valid(std::string_view candidate) {
    if (candidate.empty()) {
        return false;
    }

    if (is_runtime_root(std::filesystem::path(candidate))) {
        tt::tt_metal::SetRootDir(std::string(candidate));
        return true;
    }

    return false;
}

void ensure_runtime_root() {
    static std::once_flag once;

    std::call_once(once, [] {
        if (const char* runtime_root = std::getenv("TT_METAL_RUNTIME_ROOT")) {
            if (set_root_dir_if_valid(runtime_root)) {
                return;
            }
            throw std::runtime_error(
                "TT_METAL_RUNTIME_ROOT must point to a directory containing tt_metal/");
        }

        if (const char* metal_home = std::getenv("TT_METAL_HOME")) {
            if (set_root_dir_if_valid(metal_home)) {
                return;
            }
            throw std::runtime_error("TT_METAL_HOME must point to a directory containing tt_metal/");
        }

#ifdef TT_METAL_DEFAULT_RUNTIME_ROOT
        if (set_root_dir_if_valid(TT_METAL_DEFAULT_RUNTIME_ROOT)) {
            return;
        }
#endif

        throw std::runtime_error(
            "unable to resolve TT-Metal runtime root; set TT_METAL_RUNTIME_ROOT to a directory containing tt_metal/");
    });
}

std::mutex& device_count_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::size_t& open_device_count() {
    static std::size_t count = 0;
    return count;
}

std::size_t& open_program_count() {
    static std::size_t count = 0;
    return count;
}

void note_device_opened() {
    std::scoped_lock lock(device_count_mutex());
    ++open_device_count();
}

void note_program_opened() {
    std::scoped_lock lock(device_count_mutex());
    ++open_program_count();
}

void register_atexit_cleanup() {
    static std::once_flag once;

    std::call_once(once, [] {
        std::atexit([] {
            try {
                tt::tt_metal::detail::ReleaseOwnership();
            } catch (...) {
            }
        });
    });
}

void maybe_release_ownership() {
    bool should_release = false;
    {
        std::scoped_lock lock(device_count_mutex());
        should_release = open_device_count() == 0 && open_program_count() == 0;
    }

    if (should_release) {
        tt::tt_metal::detail::ReleaseOwnership();
    }
}

void note_device_closed() {
    std::scoped_lock lock(device_count_mutex());
    if (open_device_count() > 0) {
        --open_device_count();
    }
}

void note_program_closed() {
    std::scoped_lock lock(device_count_mutex());
    if (open_program_count() > 0) {
        --open_program_count();
    }
}

}  // namespace

DeviceHandle::DeviceHandle(tt::tt_metal::IDevice* device, std::int32_t device_id) noexcept :
    device_(device), device_id_(device_id) {}

ProgramHandle::ProgramHandle(tt::tt_metal::Program&& program) noexcept :
    program_(std::make_unique<tt::tt_metal::Program>(std::move(program))) {
    note_program_opened();
}

ProgramHandle::~ProgramHandle() {
    try {
        if (program_ != nullptr) {
            program_.reset();
            note_program_closed();
            maybe_release_ownership();
        }
    } catch (...) {
    }
}

DeviceHandle::~DeviceHandle() {
    try {
        close();
    } catch (...) {
    }
}

bool DeviceHandle::close() {
    if (device_ == nullptr) {
        return false;
    }

    auto* device = device_;
    device_ = nullptr;
    const bool closed = tt::tt_metal::CloseDevice(device);
    note_device_closed();
    maybe_release_ownership();
    return closed;
}

bool DeviceHandle::is_open() const noexcept {
    return device_ != nullptr;
}

std::int32_t DeviceHandle::device_id() const noexcept {
    return device_id_;
}

std::uint64_t ProgramHandle::runtime_id() const {
    return program_->get_runtime_id();
}

void ProgramHandle::set_runtime_id(std::uint64_t runtime_id) {
    program_->set_runtime_id(runtime_id);
}

std::unique_ptr<DeviceHandle> create_device(std::int32_t device_id) {
    ensure_runtime_root();

    if (device_id < 0) {
        throw std::invalid_argument("device_id must be non-negative");
    }

    const auto available = tt::tt_metal::GetNumAvailableDevices();
    if (static_cast<std::size_t>(device_id) >= available) {
        throw std::out_of_range("device_id is outside the available TT-Metal device range");
    }

    auto* device = tt::tt_metal::CreateDevice(static_cast<tt::ChipId>(device_id));
    if (device == nullptr) {
        throw std::runtime_error("tt::tt_metal::CreateDevice returned a null device");
    }

    register_atexit_cleanup();
    note_device_opened();
    return std::make_unique<DeviceHandle>(device, device_id);
}

std::unique_ptr<ProgramHandle> create_program() {
    ensure_runtime_root();
    register_atexit_cleanup();
    return std::make_unique<ProgramHandle>(tt::tt_metal::CreateProgram());
}

std::size_t get_num_available_devices() {
    ensure_runtime_root();
    const auto available = tt::tt_metal::GetNumAvailableDevices();
    register_atexit_cleanup();
    return available;
}

std::size_t get_num_pcie_devices() {
    ensure_runtime_root();
    const auto pcie = tt::tt_metal::GetNumPCIeDevices();
    register_atexit_cleanup();
    return pcie;
}

}  // namespace tt_metal_cxx
