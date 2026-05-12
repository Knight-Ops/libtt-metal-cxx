#[allow(clippy::module_inception)]
#[cxx::bridge(namespace = "tt_metal_cxx")]
pub(crate) mod ffi {
    struct CoreRangeRepr {
        start_x: u32,
        start_y: u32,
        end_x: u32,
        end_y: u32,
    }

    struct InterleavedBufferConfigRepr {
        size: u64,
        page_size: u64,
        buffer_type: u8,
    }

    struct ShardedBufferConfigRepr {
        size: u64,
        page_size: u64,
        buffer_type: u8,
        buffer_layout: u8,
        shard_orientation: u8,
        shard_shape_height: u32,
        shard_shape_width: u32,
        page_shape_height: u32,
        page_shape_width: u32,
        tensor2d_shape_height: u32,
        tensor2d_shape_width: u32,
    }

    struct BufferCreateOptionsRepr {
        has_address: bool,
        address: u64,
        has_sub_device_id: bool,
        sub_device_id: u8,
    }

    struct BufferInfoRepr {
        is_allocated: bool,
        address: u32,
        size: u64,
        page_size: u64,
        buffer_type: u8,
        buffer_layout: u8,
        has_sub_device_id: bool,
        sub_device_id: u8,
    }

    struct ShardSpecBufferMetadataRepr {
        shard_shape_height: u32,
        shard_shape_width: u32,
        shard_orientation: u8,
        page_shape_height: u32,
        page_shape_width: u32,
        tensor2d_shape_height: u32,
        tensor2d_shape_width: u32,
    }

    struct CircularBufferMetadataRepr {
        total_size: u32,
        has_globally_allocated_address: bool,
        globally_allocated_address: u32,
        dynamic_cb: bool,
        max_size: u32,
        buffer_size: u32,
        address_offset: u32,
    }

    struct CircularBufferIndexConfigRepr {
        buffer_index: u8,
        is_remote: bool,
        has_data_format: bool,
        data_format: u8,
        has_page_size: bool,
        page_size: u32,
        has_tile: bool,
        tile_height: u32,
        tile_width: u32,
        tile_transpose: bool,
    }

    unsafe extern "C++" {
        include!("libtt-metal-cxx/include/tt_metal_cxx.hpp");

        type DeviceHandle;
        type BufferHandle;
        type CircularBufferConfigHandle;
        type ProgramHandle;
        type ComputeKernelConfigHandle;
        type DataMovementKernelConfigHandle;
        type MeshBufferHandle;
        type MeshDeviceHandle;
        type MeshWorkloadHandle;

        fn throw_invalid_argument(message: &str) -> Result<()>;
        fn create_device(device_id: i32) -> Result<UniquePtr<DeviceHandle>>;
        fn create_interleaved_buffer(
            device: &DeviceHandle,
            config: &InterleavedBufferConfigRepr,
            options: &BufferCreateOptionsRepr,
        ) -> Result<UniquePtr<BufferHandle>>;
        fn create_sharded_buffer(
            device: &DeviceHandle,
            config: &ShardedBufferConfigRepr,
            core_ranges: &[CoreRangeRepr],
            options: &BufferCreateOptionsRepr,
        ) -> Result<UniquePtr<BufferHandle>>;
        fn create_circular_buffer_config(total_size: u32) -> UniquePtr<CircularBufferConfigHandle>;
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

        fn info(self: &BufferHandle) -> BufferInfoRepr;
        fn deallocate(self: Pin<&mut BufferHandle>) -> Result<bool>;
        fn has_shard_spec(self: &BufferHandle) -> bool;
        fn shard_spec_metadata(self: &BufferHandle) -> Result<ShardSpecBufferMetadataRepr>;
        fn shard_spec_core_ranges(self: &BufferHandle) -> Result<Vec<CoreRangeRepr>>;

        fn set_total_size(self: Pin<&mut CircularBufferConfigHandle>, total_size: u32);
        fn set_address_offset(self: Pin<&mut CircularBufferConfigHandle>, offset: u32);
        fn set_globally_allocated_address(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer: &BufferHandle,
        ) -> Result<()>;
        fn set_globally_allocated_address_and_total_size(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer: &BufferHandle,
            total_size: u32,
        ) -> Result<()>;
        fn set_index_data_format(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer_index: u8,
            remote: bool,
            data_format: u8,
        );
        fn set_index_total_size(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer_index: u8,
            remote: bool,
            total_size: u32,
        );
        fn set_index_page_size(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer_index: u8,
            remote: bool,
            page_size: u32,
        );
        fn set_index_tile(
            self: Pin<&mut CircularBufferConfigHandle>,
            buffer_index: u8,
            remote: bool,
            tile_height: u32,
            tile_width: u32,
            tile_transpose: bool,
        ) -> Result<()>;

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
        fn assign_global_buffer(self: Pin<&mut ProgramHandle>, buffer: &BufferHandle)
        -> Result<()>;
        fn create_circular_buffer(
            self: Pin<&mut ProgramHandle>,
            core_ranges: &[CoreRangeRepr],
            config: &CircularBufferConfigHandle,
        ) -> Result<usize>;
        fn get_circular_buffer_metadata(
            self: &ProgramHandle,
            cb_handle: usize,
        ) -> Result<CircularBufferMetadataRepr>;
        fn get_circular_buffer_indices(
            self: &ProgramHandle,
            cb_handle: usize,
        ) -> Result<Vec<CircularBufferIndexConfigRepr>>;
        fn update_circular_buffer_total_size(
            self: Pin<&mut ProgramHandle>,
            cb_handle: usize,
            total_size: u32,
        ) -> Result<()>;
        fn update_circular_buffer_page_size(
            self: Pin<&mut ProgramHandle>,
            cb_handle: usize,
            buffer_index: u8,
            page_size: u32,
        ) -> Result<()>;
        fn update_dynamic_circular_buffer_address(
            self: Pin<&mut ProgramHandle>,
            cb_handle: usize,
            buffer: &BufferHandle,
        ) -> Result<()>;
        fn update_dynamic_circular_buffer_address_with_offset(
            self: Pin<&mut ProgramHandle>,
            cb_handle: usize,
            buffer: &BufferHandle,
            address_offset: u32,
        ) -> Result<()>;
        fn update_dynamic_circular_buffer_address_and_total_size(
            self: Pin<&mut ProgramHandle>,
            cb_handle: usize,
            buffer: &BufferHandle,
            total_size: u32,
        ) -> Result<()>;
        fn create_semaphore(
            self: Pin<&mut ProgramHandle>,
            core_ranges: &[CoreRangeRepr],
            initial_value: u32,
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

        fn tilize(data: &[u8], m: u32, n: u32, elem_size: u32) -> Result<Vec<u8>>;
        fn untilize(data: &[u8], m: u32, n: u32, elem_size: u32) -> Result<Vec<u8>>;

        fn create_replicated_mesh_buffer(
            mesh_device: &MeshDeviceHandle,
            size_bytes: u64,
            page_size: u64,
            buffer_type: u8,
        ) -> Result<UniquePtr<MeshBufferHandle>>;
        fn address(self: &MeshBufferHandle) -> u32;
        fn size(self: &MeshBufferHandle) -> u64;
        fn is_allocated(self: &MeshBufferHandle) -> bool;

        fn write_mesh_buffer(
            self: &MeshDeviceHandle,
            buffer: &MeshBufferHandle,
            data: &[u8],
        ) -> Result<()>;
        fn read_mesh_buffer(
            self: &MeshDeviceHandle,
            buffer: &MeshBufferHandle,
            data: &mut [u8],
        ) -> Result<()>;
    }
}
