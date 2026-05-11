#include "tt_metal_cxx/device.hpp"

#include <stdexcept>

#include <tt-metalium/host_api.hpp>

#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx {

DeviceHandle::DeviceHandle(tt::tt_metal::IDevice* device, std::int32_t device_id) noexcept :
    device_(device), device_id_(device_id) {}

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
    detail::note_device_closed();
    detail::maybe_release_ownership();
    return closed;
}

bool DeviceHandle::is_open() const noexcept {
    return device_ != nullptr;
}

std::int32_t DeviceHandle::device_id() const noexcept {
    return device_id_;
}

std::unique_ptr<DeviceHandle> create_device(std::int32_t device_id) {
    detail::ensure_runtime_root();

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

    detail::register_atexit_cleanup();
    detail::note_device_opened();
    return std::make_unique<DeviceHandle>(device, device_id);
}

std::size_t get_num_available_devices() {
    detail::ensure_runtime_root();
    const auto available = tt::tt_metal::GetNumAvailableDevices();
    detail::register_atexit_cleanup();
    return available;
}

std::size_t get_num_pcie_devices() {
    detail::ensure_runtime_root();
    const auto pcie = tt::tt_metal::GetNumPCIeDevices();
    detail::register_atexit_cleanup();
    return pcie;
}

}  // namespace tt_metal_cxx
