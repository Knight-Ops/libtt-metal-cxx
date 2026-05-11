#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt::tt_metal {
class IDevice;
}

namespace tt_metal_cxx {

class DeviceHandle {
public:
    DeviceHandle(tt::tt_metal::IDevice* device, std::int32_t device_id) noexcept;
    ~DeviceHandle();

    DeviceHandle(const DeviceHandle&) = delete;
    DeviceHandle& operator=(const DeviceHandle&) = delete;
    DeviceHandle(DeviceHandle&&) = delete;
    DeviceHandle& operator=(DeviceHandle&&) = delete;

    bool close();
    bool is_open() const noexcept;
    std::int32_t device_id() const noexcept;

private:
    tt::tt_metal::IDevice* device_;
    std::int32_t device_id_;
};

std::unique_ptr<DeviceHandle> create_device(std::int32_t device_id);
std::size_t get_num_available_devices();
std::size_t get_num_pcie_devices();

}  // namespace tt_metal_cxx
