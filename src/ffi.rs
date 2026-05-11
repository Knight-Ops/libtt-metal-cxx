#[allow(clippy::module_inception)]
#[cxx::bridge(namespace = "tt_metal_cxx")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("libtt-metal-cxx/include/tt_metal_cxx.hpp");

        type DeviceHandle;
        type ProgramHandle;
        type ComputeKernelConfigHandle;
        type DataMovementKernelConfigHandle;
        type MeshDeviceHandle;
        type MeshWorkloadHandle;

        fn create_device(device_id: i32) -> Result<UniquePtr<DeviceHandle>>;
        fn create_program() -> UniquePtr<ProgramHandle>;
        fn create_compute_kernel_config() -> UniquePtr<ComputeKernelConfigHandle>;
        fn create_data_movement_kernel_config() -> UniquePtr<DataMovementKernelConfigHandle>;
        fn create_reader_data_movement_kernel_config()
        -> Result<UniquePtr<DataMovementKernelConfigHandle>>;
        fn create_writer_data_movement_kernel_config()
        -> Result<UniquePtr<DataMovementKernelConfigHandle>>;
        fn create_unit_mesh(device_id: i32) -> Result<UniquePtr<MeshDeviceHandle>>;
        fn create_mesh_workload() -> UniquePtr<MeshWorkloadHandle>;

        fn close(self: Pin<&mut DeviceHandle>) -> Result<bool>;
        fn is_open(self: &DeviceHandle) -> bool;
        fn device_id(self: &DeviceHandle) -> i32;

        fn runtime_id(self: &ProgramHandle) -> u64;
        fn set_runtime_id(self: Pin<&mut ProgramHandle>, runtime_id: u64);
        fn set_runtime_args(
            self: Pin<&mut ProgramHandle>,
            kernel_id: u32,
            core_x: u32,
            core_y: u32,
            runtime_args: &[u32],
        ) -> Result<()>;
        fn get_runtime_args(
            self: &ProgramHandle,
            kernel_id: u32,
            core_x: u32,
            core_y: u32,
        ) -> Result<Vec<u32>>;
        fn set_common_runtime_args(
            self: Pin<&mut ProgramHandle>,
            kernel_id: u32,
            runtime_args: &[u32],
        ) -> Result<()>;
        fn get_common_runtime_args(self: &ProgramHandle, kernel_id: u32) -> Result<Vec<u32>>;
        fn create_compute_kernel(
            self: Pin<&mut ProgramHandle>,
            file_name: &str,
            core_x: u32,
            core_y: u32,
        ) -> Result<u32>;
        fn create_compute_kernel_from_string(
            self: Pin<&mut ProgramHandle>,
            kernel_src_code: &str,
            core_x: u32,
            core_y: u32,
        ) -> Result<u32>;
        fn create_compute_kernel_with_config(
            self: Pin<&mut ProgramHandle>,
            file_name: &str,
            core_x: u32,
            core_y: u32,
            config: &ComputeKernelConfigHandle,
        ) -> Result<u32>;
        fn create_compute_kernel_from_string_with_config(
            self: Pin<&mut ProgramHandle>,
            kernel_src_code: &str,
            core_x: u32,
            core_y: u32,
            config: &ComputeKernelConfigHandle,
        ) -> Result<u32>;
        fn create_data_movement_kernel(
            self: Pin<&mut ProgramHandle>,
            file_name: &str,
            core_x: u32,
            core_y: u32,
            processor: u8,
            noc: u8,
        ) -> Result<u32>;
        fn create_data_movement_kernel_from_string(
            self: Pin<&mut ProgramHandle>,
            kernel_src_code: &str,
            core_x: u32,
            core_y: u32,
            processor: u8,
            noc: u8,
        ) -> Result<u32>;
        fn create_data_movement_kernel_with_config(
            self: Pin<&mut ProgramHandle>,
            file_name: &str,
            core_x: u32,
            core_y: u32,
            config: &DataMovementKernelConfigHandle,
        ) -> Result<u32>;
        fn create_data_movement_kernel_from_string_with_config(
            self: Pin<&mut ProgramHandle>,
            kernel_src_code: &str,
            core_x: u32,
            core_y: u32,
            config: &DataMovementKernelConfigHandle,
        ) -> Result<u32>;

        fn set_math_fidelity(self: Pin<&mut ComputeKernelConfigHandle>, math_fidelity: u8);
        fn set_fp32_dest_acc_en(self: Pin<&mut ComputeKernelConfigHandle>, enabled: bool);
        fn set_dst_full_sync_en(self: Pin<&mut ComputeKernelConfigHandle>, enabled: bool);
        fn fill_unpack_to_dest_modes(self: Pin<&mut ComputeKernelConfigHandle>, mode: u8);
        fn set_bfp8_pack_precise(self: Pin<&mut ComputeKernelConfigHandle>, enabled: bool);
        fn set_math_approx_mode(self: Pin<&mut ComputeKernelConfigHandle>, enabled: bool);
        fn add_compile_arg(self: Pin<&mut ComputeKernelConfigHandle>, arg: u32);
        fn add_define(self: Pin<&mut ComputeKernelConfigHandle>, key: &str, value: &str);
        fn add_named_compile_arg(self: Pin<&mut ComputeKernelConfigHandle>, key: &str, value: u32);
        fn set_opt_level(self: Pin<&mut ComputeKernelConfigHandle>, opt_level: u8);

        fn set_processor(self: Pin<&mut DataMovementKernelConfigHandle>, processor: u8);
        fn set_noc(self: Pin<&mut DataMovementKernelConfigHandle>, noc: u8);
        fn set_noc_mode(self: Pin<&mut DataMovementKernelConfigHandle>, noc_mode: u8);
        fn add_compile_arg(self: Pin<&mut DataMovementKernelConfigHandle>, arg: u32);
        fn add_define(self: Pin<&mut DataMovementKernelConfigHandle>, key: &str, value: &str);
        fn add_named_compile_arg(
            self: Pin<&mut DataMovementKernelConfigHandle>,
            key: &str,
            value: u32,
        );
        fn set_opt_level(self: Pin<&mut DataMovementKernelConfigHandle>, opt_level: u8);

        fn close(self: Pin<&mut MeshDeviceHandle>) -> Result<bool>;
        fn is_open(self: &MeshDeviceHandle) -> bool;
        fn device_id(self: &MeshDeviceHandle) -> i32;
        fn num_devices(self: &MeshDeviceHandle) -> Result<usize>;
        fn num_rows(self: &MeshDeviceHandle) -> Result<usize>;
        fn num_cols(self: &MeshDeviceHandle) -> Result<usize>;
        fn enqueue_workload(
            self: &MeshDeviceHandle,
            workload: Pin<&mut MeshWorkloadHandle>,
            blocking: bool,
        ) -> Result<()>;

        fn add_program_to_full_mesh(
            self: Pin<&mut MeshWorkloadHandle>,
            mesh_device: &MeshDeviceHandle,
            program: UniquePtr<ProgramHandle>,
        ) -> Result<()>;
        fn program_count(self: &MeshWorkloadHandle) -> usize;

        fn get_num_available_devices() -> Result<usize>;
        fn get_num_pcie_devices() -> Result<usize>;
    }
}
