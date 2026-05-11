#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt::tt_metal {
class IDevice;
class Program;
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

private:
    std::unique_ptr<tt::tt_metal::Program> program_;
};

std::unique_ptr<DeviceHandle> create_device(std::int32_t device_id);
std::unique_ptr<ProgramHandle> create_program();
std::size_t get_num_available_devices();
std::size_t get_num_pcie_devices();

}  // namespace tt_metal_cxx
