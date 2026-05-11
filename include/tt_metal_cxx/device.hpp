#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt::tt_metal {
class IDevice;
}

namespace tt_metal_cxx::detail {
struct DeviceContext;
}

namespace tt_metal_cxx {

class DeviceHandle {
public:
    explicit DeviceHandle(std::shared_ptr<detail::DeviceContext> context) noexcept;
    ~DeviceHandle();

    DeviceHandle(const DeviceHandle&) = delete;
    DeviceHandle& operator=(const DeviceHandle&) = delete;
    DeviceHandle(DeviceHandle&&) = delete;
    DeviceHandle& operator=(DeviceHandle&&) = delete;

    bool close();
    bool is_open() const noexcept;
    std::int32_t device_id() const noexcept;
    tt::tt_metal::IDevice* raw_device() const noexcept;
    std::shared_ptr<detail::DeviceContext> context() const noexcept;

private:
    std::shared_ptr<detail::DeviceContext> context_;
};

std::unique_ptr<DeviceHandle> create_device(std::int32_t device_id);
std::size_t get_num_available_devices();
std::size_t get_num_pcie_devices();

}  // namespace tt_metal_cxx
