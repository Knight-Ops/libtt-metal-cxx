#include "tt_metal_cxx/device.hpp"

#include <memory>
#include <stdexcept>

#include <tt-metalium/host_api.hpp>

#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx::detail {

struct DeviceContext {
    DeviceContext(tt::tt_metal::IDevice* device_ptr, std::int32_t chip_id) noexcept :
        device(device_ptr), device_id(chip_id) {
        note_device_opened();
    }

    ~DeviceContext() {
        try {
            release();
        } catch (...) {
        }
    }

    bool release() {
        if (device == nullptr) {
            return false;
        }

        auto* raw_device = device;
        device = nullptr;
        const bool closed = tt::tt_metal::CloseDevice(raw_device);
        note_device_closed();
        maybe_release_ownership();
        return closed;
    }

    bool is_open() const noexcept { return device != nullptr; }

    tt::tt_metal::IDevice* device;
    std::int32_t device_id;
};

}  // namespace tt_metal_cxx::detail

namespace tt_metal_cxx {

DeviceHandle::DeviceHandle(std::shared_ptr<detail::DeviceContext> context) noexcept : context_(std::move(context)) {}

DeviceHandle::~DeviceHandle() {
    try {
        close();
    } catch (...) {
    }
}

bool DeviceHandle::close() {
    if (context_ == nullptr) {
        return false;
    }

    auto context = std::move(context_);
    context_.reset();
    if (context.use_count() == 1) {
        return context->release();
    }
    return true;
}

bool DeviceHandle::is_open() const noexcept {
    return context_ != nullptr;
}

std::int32_t DeviceHandle::device_id() const noexcept {
    return context_ != nullptr ? context_->device_id : -1;
}

tt::tt_metal::IDevice* DeviceHandle::raw_device() const noexcept {
    return context_ != nullptr ? context_->device : nullptr;
}

std::shared_ptr<detail::DeviceContext> DeviceHandle::context() const noexcept {
    return context_;
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
    return std::make_unique<DeviceHandle>(std::make_shared<detail::DeviceContext>(device, device_id));
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
