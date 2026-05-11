#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}

namespace tt_metal_cxx {

class ProgramHandle;

class MeshWorkloadHandle;

class MeshDeviceHandle {
public:
    MeshDeviceHandle(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, std::int32_t device_id) noexcept;
    ~MeshDeviceHandle();

    MeshDeviceHandle(const MeshDeviceHandle&) = delete;
    MeshDeviceHandle& operator=(const MeshDeviceHandle&) = delete;
    MeshDeviceHandle(MeshDeviceHandle&&) = delete;
    MeshDeviceHandle& operator=(MeshDeviceHandle&&) = delete;

    bool close();
    bool is_open() const noexcept;
    std::int32_t device_id() const noexcept;
    std::size_t num_devices() const;
    std::size_t num_rows() const;
    std::size_t num_cols() const;
    void enqueue_workload(MeshWorkloadHandle& workload, bool blocking) const;

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    std::int32_t device_id_;

    friend class MeshWorkloadHandle;
};

class MeshWorkloadHandle {
public:
    MeshWorkloadHandle();
    ~MeshWorkloadHandle();

    MeshWorkloadHandle(const MeshWorkloadHandle&) = delete;
    MeshWorkloadHandle& operator=(const MeshWorkloadHandle&) = delete;
    MeshWorkloadHandle(MeshWorkloadHandle&&) = delete;
    MeshWorkloadHandle& operator=(MeshWorkloadHandle&&) = delete;

    void add_program_to_full_mesh(const MeshDeviceHandle& mesh_device, std::unique_ptr<ProgramHandle> program);
    std::size_t program_count() const;

private:
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload_;
    std::size_t owned_program_count_ = 0;

    friend class MeshDeviceHandle;
};

std::unique_ptr<MeshDeviceHandle> create_unit_mesh(std::int32_t device_id);
std::unique_ptr<MeshWorkloadHandle> create_mesh_workload();

}  // namespace tt_metal_cxx
