use std::env;
use std::sync::{Mutex, MutexGuard, OnceLock};

use libtt_metal_cxx::{
    Buffer, BufferCreateOptions, BufferType, CircularBufferConfig, ComputeKernelConfig, CoreRange,
    CoreRangeSet, DataFormat, DataMovementKernelConfig, DataMovementProcessor, Device,
    InterleavedBufferConfig, KernelBuildOptLevel, LogicalCore, MathFidelity, MeshDevice,
    MeshWorkload, Noc, Program, ShardOrientation, ShardSpecBuffer, ShardedBufferConfig,
    TensorMemoryLayout, available_device_count, pcie_device_count, query_devices,
};

fn device_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn hardware_tests_enabled() -> bool {
    env::var_os("TT_METAL_RUN_HARDWARE_TESTS").is_some()
}

fn cleanup_query_context_if_needed(available: usize) {
    if available == 0 {
        return;
    }

    let mut device = Device::create(0).expect("device 0 should open for query cleanup");
    assert!(device.close().expect("query cleanup close should succeed"));
}

fn packaged_compute_blank_kernel() -> &'static str {
    "tt_metal/kernels/compute/blank.cpp"
}

fn packaged_dataflow_blank_kernel() -> &'static str {
    "tt_metal/kernels/dataflow/blank.cpp"
}

fn inline_compute_kernel_source() -> &'static str {
    r#"
    #include "api/compute/compute_kernel_api.h"

    void kernel_main() {}
    "#
}

fn inline_dataflow_kernel_source() -> &'static str {
    r#"
    #include "api/dataflow/dataflow_api.h"

    void kernel_main() {}
    "#
}

fn single_worker_core() -> LogicalCore {
    LogicalCore::new(0, 0)
}

fn second_worker_core() -> LogicalCore {
    LogicalCore::new(0, 1)
}

fn single_core_range_set() -> CoreRangeSet {
    CoreRangeSet::from_core(single_worker_core())
}

fn multi_range_set() -> CoreRangeSet {
    CoreRangeSet::from_ranges([
        CoreRange::from_core(single_worker_core()),
        CoreRange::from_core(second_worker_core()),
    ])
}

fn dram_buffer_config(size: u64) -> InterleavedBufferConfig {
    InterleavedBufferConfig::new(size, size, BufferType::Dram)
}

fn l1_buffer_config(size: u64) -> InterleavedBufferConfig {
    InterleavedBufferConfig::new(size, size, BufferType::L1)
}

fn single_core_shard_spec() -> ShardSpecBuffer {
    ShardSpecBuffer::new(
        single_core_range_set(),
        [1, 1],
        ShardOrientation::RowMajor,
        [1, 1],
        [1, 1],
    )
}

#[test]
fn query_devices_is_self_consistent() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let counts = query_devices().expect("query_devices should succeed");
    assert!(counts.pcie <= counts.available);
    assert_eq!(
        counts.available,
        available_device_count().expect("available device count should succeed")
    );
    assert_eq!(
        counts.pcie,
        pcie_device_count().expect("PCIe device count should succeed")
    );
    cleanup_query_context_if_needed(counts.available);
}

#[test]
fn create_device_rejects_negative_ids() {
    let error = match Device::create(-1) {
        Ok(_) => panic!("negative device id should be rejected"),
        Err(error) => error,
    };
    assert!(
        error.what().contains("non-negative"),
        "unexpected error message: {}",
        error.what()
    );
}

#[test]
fn create_device_rejects_out_of_range_ids() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let invalid_device_id =
        available_device_count().expect("available device count should succeed") as i32;
    let error = match Device::create(invalid_device_id) {
        Ok(_) => panic!("out-of-range device id should be rejected before calling TT-Metal"),
        Err(error) => error,
    };
    assert!(
        error.what().contains("outside the available"),
        "unexpected error message: {}",
        error.what()
    );
    cleanup_query_context_if_needed(invalid_device_id as usize);
}

#[test]
fn create_and_close_device_round_trip() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    if available_device_count().expect("available device count should succeed") == 0 {
        return;
    }

    let mut device = Device::create(0).expect("device 0 should open");
    assert!(device.is_open());
    assert_eq!(device.device_id(), Some(0));
    assert!(
        device.close().expect("close should succeed"),
        "close should return true for an open device"
    );
    assert!(!device.is_open());
    assert!(
        !device.close().expect("second close should succeed"),
        "close should become a no-op after the device is closed"
    );
}

#[test]
fn drop_closes_device_for_reopen() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    if available_device_count().expect("available device count should succeed") == 0 {
        return;
    }

    {
        let device = Device::create(0).expect("device 0 should open");
        assert!(device.is_open());
    }

    let mut reopened = Device::create(0).expect("device 0 should reopen after drop");
    assert!(reopened.is_open());
    assert!(reopened.close().expect("close should succeed"));
}

#[test]
fn create_program_starts_with_default_runtime_id() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let program = Program::create();
    assert_eq!(program.runtime_id(), Some(0));
}

#[test]
fn create_program_round_trips_runtime_id() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut program = Program::create();
    program.set_runtime_id(42);
    assert_eq!(program.runtime_id(), Some(42));
}

#[test]
fn create_program_instances_are_independent() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut first = Program::create();
    let second = Program::create();

    first.set_runtime_id(7);

    assert_eq!(first.runtime_id(), Some(7));
    assert_eq!(second.runtime_id(), Some(0));
}

#[test]
fn create_unit_mesh_exposes_basic_shape() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    assert!(mesh.is_open());
    assert_eq!(mesh.device_id(), Some(0));
    assert_eq!(mesh.num_devices().expect("num_devices should succeed"), 1);
    assert_eq!(mesh.num_rows().expect("num_rows should succeed"), 1);
    assert_eq!(mesh.num_cols().expect("num_cols should succeed"), 1);
    assert!(mesh.close().expect("mesh close should succeed"));
    assert!(!mesh.is_open());
}

#[test]
fn mesh_workload_accepts_program_for_full_mesh() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::create();
    let program = Program::create();

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept a full-mesh program");
    assert_eq!(workload.program_count(), 1);
    mesh.enqueue_workload(&mut workload, true)
        .expect("mesh workload should enqueue successfully");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn program_accepts_custom_kernel_configs_and_enqueues() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::create();
    let mut program = Program::create();
    let core = LogicalCore::new(0, 0);
    let mut compute_config = ComputeKernelConfig::new();
    compute_config
        .set_math_fidelity(MathFidelity::HiFi4)
        .set_opt_level(KernelBuildOptLevel::O3)
        .add_compile_args([11, 22, 33])
        .add_define("RUST_KERNEL_TEST_DEFINE", "1")
        .add_named_compile_arg("rust_kernel_test_arg", 7);
    let mut dataflow_config =
        DataMovementKernelConfig::reader().expect("reader data movement config should initialize");
    dataflow_config
        .set_processor(DataMovementProcessor::Riscv1)
        .add_compile_args([5, 6])
        .add_define("RUST_DM_KERNEL_TEST_DEFINE", "1")
        .add_named_compile_arg("rust_dm_kernel_test_arg", 9)
        .set_opt_level(KernelBuildOptLevel::O2);

    let compute_kernel_id = program
        .create_compute_kernel_with_config(packaged_compute_blank_kernel(), core, &compute_config)
        .expect("program should accept a configured blank compute kernel");
    let dataflow_kernel_id = program
        .create_data_movement_kernel_with_config(
            packaged_dataflow_blank_kernel(),
            core,
            &dataflow_config,
        )
        .expect("program should accept a configured blank data movement kernel");

    assert_ne!(compute_kernel_id, dataflow_kernel_id);

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept a program containing kernels");
    mesh.enqueue_workload(&mut workload, true)
        .expect("kernel workload should enqueue successfully");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn program_accepts_inline_kernel_sources_and_enqueues() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::create();
    let mut program = Program::create();
    let core = LogicalCore::new(0, 0);
    let mut compute_config = ComputeKernelConfig::new();
    compute_config
        .set_math_fidelity(MathFidelity::HiFi4)
        .set_opt_level(KernelBuildOptLevel::O3)
        .add_compile_arg(17);
    let mut dataflow_config =
        DataMovementKernelConfig::reader().expect("reader data movement config should initialize");
    dataflow_config
        .set_processor(DataMovementProcessor::Riscv1)
        .add_compile_arg(23)
        .set_opt_level(KernelBuildOptLevel::O2);

    let compute_kernel_id = program
        .create_compute_kernel_from_string_with_config(
            inline_compute_kernel_source(),
            core,
            &compute_config,
        )
        .expect("program should accept inline compute kernel source");
    let dataflow_kernel_id = program
        .create_data_movement_kernel_from_string_with_config(
            inline_dataflow_kernel_source(),
            core,
            &dataflow_config,
        )
        .expect("program should accept inline data movement kernel source");

    assert_ne!(compute_kernel_id, dataflow_kernel_id);

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept a program containing inline-source kernels");
    mesh.enqueue_workload(&mut workload, true)
        .expect("inline-source kernel workload should enqueue successfully");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn program_runtime_args_round_trip_and_update() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::create();
    let mut program = Program::create();
    let core = LogicalCore::new(0, 0);

    let compute_kernel_id = program
        .create_compute_kernel(packaged_compute_blank_kernel(), core)
        .expect("program should accept a blank compute kernel");
    let dataflow_kernel_id = program
        .create_data_movement_kernel(
            packaged_dataflow_blank_kernel(),
            core,
            DataMovementProcessor::Riscv1,
            Noc::Riscv1Default,
        )
        .expect("program should accept a blank data movement kernel");

    program
        .set_runtime_args(compute_kernel_id, core, &[1, 2, 3, 4])
        .expect("compute runtime args should set");
    assert_eq!(
        program
            .runtime_args(compute_kernel_id, core)
            .expect("compute runtime args should read back"),
        vec![1, 2, 3, 4]
    );
    program
        .set_runtime_args(compute_kernel_id, core, &[9, 10, 11, 12])
        .expect("compute runtime args should update");
    assert_eq!(
        program
            .runtime_args(compute_kernel_id, core)
            .expect("updated compute runtime args should read back"),
        vec![9, 10, 11, 12]
    );
    let error = program
        .set_runtime_args(compute_kernel_id, core, &[42, 43])
        .expect_err("changing compute runtime arg length should be rejected");
    assert!(
        error.what().contains("cannot be modified from 4 to 2"),
        "unexpected error message: {}",
        error.what()
    );
    program
        .set_common_runtime_args(compute_kernel_id, &[100, 200])
        .expect("compute common runtime args should set");
    assert_eq!(
        program
            .common_runtime_args(compute_kernel_id)
            .expect("compute common runtime args should read back"),
        vec![100, 200]
    );

    program
        .set_runtime_args(dataflow_kernel_id, core, &[7, 8, 9])
        .expect("data movement runtime args should set");
    assert_eq!(
        program
            .runtime_args(dataflow_kernel_id, core)
            .expect("data movement runtime args should read back"),
        vec![7, 8, 9]
    );
    program
        .set_common_runtime_args(dataflow_kernel_id, &[300])
        .expect("data movement common runtime args should set");
    assert_eq!(
        program
            .common_runtime_args(dataflow_kernel_id)
            .expect("data movement common runtime args should read back"),
        vec![300]
    );

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept a program containing runtime args");
    mesh.enqueue_workload(&mut workload, true)
        .expect("runtime-args workload should enqueue successfully");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn interleaved_buffers_round_trip_metadata_and_deallocate() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");

    let mut dram_buffer = Buffer::create_interleaved(
        &device,
        dram_buffer_config(4096),
        BufferCreateOptions::new(),
    )
    .expect("DRAM buffer should allocate");
    assert!(dram_buffer.is_allocated());
    assert_eq!(dram_buffer.size(), 4096);
    assert_eq!(dram_buffer.page_size(), 4096);
    assert_eq!(dram_buffer.buffer_type(), BufferType::Dram);
    assert_eq!(dram_buffer.buffer_layout(), TensorMemoryLayout::Interleaved);
    let dram_address = dram_buffer.address();
    assert!(dram_address > 0);
    assert!(dram_buffer.deallocate().expect("deallocate should succeed"));
    assert!(!dram_buffer.is_allocated());
    assert!(
        !dram_buffer
            .deallocate()
            .expect("second deallocate should be a no-op"),
        "second deallocate should return false"
    );

    let l1_buffer =
        Buffer::create_interleaved(&device, l1_buffer_config(2048), BufferCreateOptions::new())
            .expect("L1 buffer should allocate");
    assert!(l1_buffer.is_allocated());
    assert_eq!(l1_buffer.buffer_type(), BufferType::L1);
    assert_eq!(l1_buffer.page_size(), 2048);
}

#[test]
fn sharded_buffer_round_trips_layout_and_shard_spec() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let config = ShardedBufferConfig::new(
        2048,
        2048,
        BufferType::L1,
        TensorMemoryLayout::HeightSharded,
        single_core_shard_spec(),
    );

    let buffer = Buffer::create_sharded(&device, &config, BufferCreateOptions::new())
        .expect("sharded L1 buffer should allocate");
    assert!(buffer.is_allocated());
    assert_eq!(buffer.buffer_layout(), TensorMemoryLayout::HeightSharded);

    let shard_spec = buffer
        .shard_spec()
        .expect("shard spec query should succeed")
        .expect("sharded buffer should expose its shard spec");
    assert_eq!(
        shard_spec.shard_spec.orientation,
        ShardOrientation::RowMajor
    );
    assert_eq!(shard_spec.shard_spec.shape, [1, 1]);
    assert_eq!(shard_spec.page_shape, [1, 1]);
    assert_eq!(shard_spec.tensor2d_shape_in_pages, [1, 1]);
    assert_eq!(shard_spec.grid.ranges(), single_core_range_set().ranges());
}

#[test]
fn interleaved_buffer_can_be_recreated_at_fixed_address() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let config = dram_buffer_config(4096);
    let mut first = Buffer::create_interleaved(&device, config, BufferCreateOptions::new())
        .expect("first DRAM buffer should allocate");
    let address = first.address();
    assert!(first.deallocate().expect("first buffer should deallocate"));

    let second = Buffer::create_interleaved(
        &device,
        config,
        BufferCreateOptions::new().with_address(u64::from(address)),
    )
    .expect("fixed-address DRAM buffer should allocate");
    assert_eq!(second.address(), address);
    assert!(second.is_allocated());
}

#[test]
fn assigned_global_buffer_survives_rust_wrapper_drop() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::create();

    let buffer = Buffer::create_interleaved(
        &device,
        dram_buffer_config(4096),
        BufferCreateOptions::new(),
    )
    .expect("DRAM buffer should allocate");
    program
        .assign_global_buffer(&buffer)
        .expect("program should take ownership of the global buffer");
    drop(buffer);

    program.set_runtime_id(99);
    assert_eq!(program.runtime_id(), Some(99));
}

#[test]
fn program_circular_buffers_and_semaphores_round_trip_and_update() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::create();
    let replacement =
        Buffer::create_interleaved(&device, l1_buffer_config(1024), BufferCreateOptions::new())
            .expect("replacement L1 buffer should allocate");

    let backing =
        Buffer::create_interleaved(&device, l1_buffer_config(1024), BufferCreateOptions::new())
            .expect("backing L1 buffer should allocate");
    let backing_address = backing.address();

    let mut cb_config = CircularBufferConfig::new(256);
    cb_config
        .index(0)
        .set_data_format(DataFormat::RawUInt8)
        .set_page_size(64)
        .set_total_size(256);
    cb_config
        .index(1)
        .set_data_format(DataFormat::RawUInt8)
        .set_page_size(64);
    cb_config
        .set_globally_allocated_address(&backing)
        .expect("circular buffer config should accept a global backing buffer");

    let cb_id = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect("program should create a circular buffer");
    drop(backing);

    let initial_snapshot = program
        .circular_buffer_config(cb_id)
        .expect("circular buffer config should read back");
    assert_eq!(initial_snapshot.total_size, 256);
    assert_eq!(
        initial_snapshot.globally_allocated_address,
        Some(backing_address)
    );
    assert_eq!(initial_snapshot.indices.len(), 2);
    assert_eq!(initial_snapshot.indices[0].buffer_index, 0);
    assert!(!initial_snapshot.indices[0].is_remote);
    assert_eq!(
        initial_snapshot.indices[0].data_format,
        Some(DataFormat::RawUInt8)
    );
    assert_eq!(initial_snapshot.indices[0].page_size, Some(64));
    assert_eq!(initial_snapshot.indices[1].buffer_index, 1);
    assert!(!initial_snapshot.indices[1].is_remote);

    let single_semaphore = program
        .create_semaphore(&single_core_range_set(), 7)
        .expect("single-range semaphore should create");
    let multi_semaphore = program
        .create_semaphore(&multi_range_set(), 11)
        .expect("multi-range semaphore should create");
    assert_ne!(single_semaphore, multi_semaphore);

    program
        .update_circular_buffer_total_size(cb_id, 512)
        .expect("total size update should succeed");
    program
        .update_circular_buffer_page_size(cb_id, 0, 128)
        .expect("page size update should succeed");
    program
        .update_dynamic_circular_buffer_address(cb_id, &replacement)
        .expect("dynamic address update should succeed");
    program
        .update_dynamic_circular_buffer_address_with_offset(cb_id, &replacement, 32)
        .expect("dynamic address+offset update should succeed");
    program
        .update_dynamic_circular_buffer_address_and_total_size(cb_id, &replacement, 512)
        .expect("dynamic address+total-size update should succeed");

    let updated_snapshot = program
        .circular_buffer_config(cb_id)
        .expect("updated circular buffer config should read back");
    assert_eq!(updated_snapshot.total_size, 512);
    assert_eq!(
        updated_snapshot.globally_allocated_address,
        Some(replacement.address())
    );
    assert_eq!(updated_snapshot.address_offset, 32);
    assert_eq!(
        updated_snapshot
            .indices
            .iter()
            .find(|index| index.buffer_index == 0)
            .and_then(|index| index.page_size),
        Some(128)
    );
}
