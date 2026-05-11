#include "tt_metal_cxx/distributed.hpp"

#include <memory>
#include <stdexcept>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "tt_metal_cxx/program.hpp"
#include "tt_metal_cxx/runtime.hpp"

namespace tt_metal_cxx {

MeshDeviceHandle::MeshDeviceHandle(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, std::int32_t device_id) noexcept :
    mesh_device_(std::move(mesh_device)), device_id_(device_id) {
    detail::note_mesh_opened();
}

MeshDeviceHandle::~MeshDeviceHandle() {
    try {
        close();
    } catch (...) {
    }
}

MeshWorkloadHandle::MeshWorkloadHandle() :
    workload_(std::make_unique<tt::tt_metal::distributed::MeshWorkload>()) {}

MeshWorkloadHandle::~MeshWorkloadHandle() {
    try {
        if (workload_ != nullptr) {
            workload_.reset();
            while (owned_program_count_ > 0) {
                detail::note_program_closed();
                --owned_program_count_;
            }
            detail::maybe_release_ownership();
        }
    } catch (...) {
    }
}

bool MeshDeviceHandle::close() {
    if (mesh_device_ == nullptr) {
        return false;
    }

    auto mesh_device = std::move(mesh_device_);
    const bool closed = mesh_device->close();
    detail::note_mesh_closed();
    detail::maybe_release_ownership();
    return closed;
}

bool MeshDeviceHandle::is_open() const noexcept {
    return mesh_device_ != nullptr;
}

std::int32_t MeshDeviceHandle::device_id() const noexcept {
    return device_id_;
}

std::size_t MeshDeviceHandle::num_devices() const {
    if (mesh_device_ == nullptr) {
        throw std::runtime_error("mesh device is closed");
    }
    return mesh_device_->num_devices();
}

std::size_t MeshDeviceHandle::num_rows() const {
    if (mesh_device_ == nullptr) {
        throw std::runtime_error("mesh device is closed");
    }
    return mesh_device_->num_rows();
}

std::size_t MeshDeviceHandle::num_cols() const {
    if (mesh_device_ == nullptr) {
        throw std::runtime_error("mesh device is closed");
    }
    return mesh_device_->num_cols();
}

void MeshDeviceHandle::enqueue_workload(MeshWorkloadHandle& workload, bool blocking) const {
    if (mesh_device_ == nullptr) {
        throw std::runtime_error("mesh device is closed");
    }
    if (workload.workload_ == nullptr) {
        throw std::runtime_error("mesh workload is invalid");
    }

    auto& mesh_cq = mesh_device_->mesh_command_queue();
    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_cq, *workload.workload_, blocking);
}

void MeshWorkloadHandle::add_program_to_full_mesh(
    const MeshDeviceHandle& mesh_device, std::unique_ptr<ProgramHandle> program) {
    if (workload_ == nullptr) {
        throw std::runtime_error("mesh workload is invalid");
    }
    if (!mesh_device.is_open()) {
        throw std::runtime_error("mesh device is closed");
    }
    if (program == nullptr || program->program_ == nullptr) {
        throw std::runtime_error("program is invalid");
    }

    workload_->add_program(
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.mesh_device_->shape()),
        std::move(*program->program_));
    program->program_.reset();
    ++owned_program_count_;
}

std::size_t MeshWorkloadHandle::program_count() const {
    return owned_program_count_;
}

std::unique_ptr<MeshDeviceHandle> create_unit_mesh(std::int32_t device_id) {
    detail::ensure_runtime_root();
    detail::register_atexit_cleanup();

    if (device_id < 0) {
        throw std::invalid_argument("device_id must be non-negative");
    }

    const auto available = tt::tt_metal::GetNumAvailableDevices();
    if (static_cast<std::size_t>(device_id) >= available) {
        throw std::out_of_range("device_id is outside the available TT-Metal device range");
    }

    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(static_cast<tt::ChipId>(device_id));
    if (mesh_device == nullptr) {
        throw std::runtime_error("tt::tt_metal::distributed::MeshDevice::create_unit_mesh returned a null mesh");
    }

    return std::make_unique<MeshDeviceHandle>(std::move(mesh_device), device_id);
}

std::unique_ptr<MeshWorkloadHandle> create_mesh_workload() {
    return std::make_unique<MeshWorkloadHandle>();
}

}  // namespace tt_metal_cxx
