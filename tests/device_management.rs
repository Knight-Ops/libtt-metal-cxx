use std::env;
use std::sync::{Mutex, MutexGuard, OnceLock};

use libtt_metal_cxx::{
    Buffer, BufferCreateOptions, BufferType, CircularBufferConfig, ComputeKernelConfig, CoreRange,
    CoreRangeSet, DataFormat, DataMovementKernelConfig, DataMovementProcessor, Device,
    InterleavedBufferConfig, KernelBuildOptLevel, LogicalCore, MathFidelity, MeshBuffer,
    MeshDevice, MeshWorkload, Noc, NocMode, Program, ShardOrientation, ShardSpecBuffer,
    ShardedBufferConfig, TensorMemoryLayout, TileConfig, UnpackToDestMode, available_device_count,
    pcie_device_count, query_devices, tilize, untilize,
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
    let program = Program::new();
    assert_eq!(program.runtime_id(), Some(0));
}

#[test]
fn create_program_round_trips_runtime_id() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut program = Program::new();
    program.set_runtime_id(42);
    assert_eq!(program.runtime_id(), Some(42));
}

#[test]
fn create_program_instances_are_independent() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut first = Program::new();
    let second = Program::new();

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
    let mut workload = MeshWorkload::new();
    let program = Program::new();

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
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
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
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
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
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
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
    let mut program = Program::new();

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
    let mut program = Program::new();
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

#[test]
fn mesh_buffer_write_read_round_trip() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");

    const TILE_SIZE: u64 = 32 * 32 * 2; // 32x32 bfloat16 tile = 2048 bytes
    const BUFFER_SIZE: u64 = TILE_SIZE; // one tile

    // Create a replicated mesh buffer in DRAM
    let buffer = MeshBuffer::create_replicated(
        &mesh,
        BUFFER_SIZE,
        TILE_SIZE, // page_size = one tile
        0,         // 0 = DRAM
    )
    .expect("mesh buffer should allocate");
    assert!(buffer.is_allocated());
    assert_eq!(buffer.size(), BUFFER_SIZE);
    assert!(buffer.address() > 0);

    // Actually, let's just use a simple pattern
    let mut input = vec![0u8; BUFFER_SIZE as usize];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    mesh.write_mesh_buffer(&buffer, &input)
        .expect("write should succeed");

    // Read back
    let mut output = vec![0u8; BUFFER_SIZE as usize];
    mesh.read_mesh_buffer(&buffer, &mut output)
        .expect("read should succeed");

    assert_eq!(
        input, output,
        "mesh buffer read/write round-trip data mismatch"
    );

    assert!(mesh.close().expect("mesh close should succeed"));
}

// ── ComputeKernelConfig: remaining setters ──

#[test]
fn compute_kernel_config_set_unpack_to_dest_mode_and_enqueue() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
    let core = LogicalCore::new(0, 0);

    let mut compute_config = ComputeKernelConfig::new();
    compute_config.set_unpack_to_dest_modes_all(UnpackToDestMode::Default);

    program
        .create_compute_kernel_with_config(packaged_compute_blank_kernel(), core, &compute_config)
        .expect("compute kernel with unpack_to_dest mode should compile");

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept program");
    mesh.enqueue_workload(&mut workload, true)
        .expect("enqueue should succeed");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn compute_kernel_config_math_fidelity_variants_compile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let core = LogicalCore::new(0, 0);

    for math_fidelity in [MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi3] {
        let mut config = ComputeKernelConfig::new();
        config.set_math_fidelity(math_fidelity);
        let mut program = Program::new();
        program
            .create_compute_kernel_with_config(packaged_compute_blank_kernel(), core, &config)
            .expect(&format!(
                "blank compute kernel should compile with {math_fidelity:?}"
            ));
    }

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn compute_kernel_config_opt_level_variants_compile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let core = LogicalCore::new(0, 0);

    for opt_level in [
        KernelBuildOptLevel::O0,
        KernelBuildOptLevel::O1,
        KernelBuildOptLevel::Os,
        KernelBuildOptLevel::Ofast,
        KernelBuildOptLevel::Oz,
    ] {
        let mut config = ComputeKernelConfig::new();
        config.set_opt_level(opt_level);
        let mut program = Program::new();
        program
            .create_compute_kernel_with_config(packaged_compute_blank_kernel(), core, &config)
            .expect(&format!(
                "blank compute kernel should compile with {opt_level:?}"
            ));
    }

    assert!(mesh.close().expect("mesh close should succeed"));
}

// ── DataMovementKernelConfig: writer / noc / noc_mode ──

#[test]
fn data_movement_kernel_config_writer_constructor_works() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut program = Program::new();
    let core = LogicalCore::new(0, 0);

    let mut writer_config =
        DataMovementKernelConfig::writer().expect("writer config should initialize");
    writer_config.set_processor(DataMovementProcessor::Riscv0);

    program
        .create_data_movement_kernel_with_config(
            packaged_dataflow_blank_kernel(),
            core,
            &writer_config,
        )
        .expect("program should accept a writer-configured data movement kernel");

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn data_movement_kernel_config_noc_and_noc_mode_compile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let core = LogicalCore::new(0, 0);

    for (noc, noc_mode) in [
        (Noc::Noc0, NocMode::Dedicated),
        (Noc::Noc1, NocMode::Dynamic),
        (Noc::Riscv0Default, NocMode::Dedicated),
    ] {
        let mut config = DataMovementKernelConfig::new();
        config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_noc(noc)
            .set_noc_mode(noc_mode);
        let mut program = Program::new();
        program
            .create_data_movement_kernel_with_config(
                packaged_dataflow_blank_kernel(),
                core,
                &config,
            )
            .expect(&format!(
                "data movement kernel should compile with {noc:?} {noc_mode:?}"
            ));
    }

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn data_movement_kernel_config_opt_level_variants_compile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let core = LogicalCore::new(0, 0);

    for opt_level in [
        KernelBuildOptLevel::O0,
        KernelBuildOptLevel::O1,
        KernelBuildOptLevel::Os,
        KernelBuildOptLevel::Ofast,
        KernelBuildOptLevel::Oz,
    ] {
        let mut config = DataMovementKernelConfig::new();
        config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(opt_level);
        let mut program = Program::new();
        program
            .create_data_movement_kernel_with_config(
                packaged_dataflow_blank_kernel(),
                core,
                &config,
            )
            .expect(&format!(
                "data movement kernel should compile with {opt_level:?}"
            ));
    }

    assert!(mesh.close().expect("mesh close should succeed"));
}

// ── Data movement kernel from string (basic 4-arg) ──

#[test]
fn program_creates_data_movement_kernel_from_string() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
    let core = LogicalCore::new(0, 0);

    let kernel_id = program
        .create_data_movement_kernel_from_string(
            inline_dataflow_kernel_source(),
            core,
            DataMovementProcessor::Riscv1,
            Noc::Riscv1Default,
        )
        .expect("program should accept inline data movement kernel source");

    program
        .set_runtime_args(kernel_id, core, &[1, 2, 3])
        .expect("runtime args should set");
    assert_eq!(
        program
            .runtime_args(kernel_id, core)
            .expect("runtime args should read back"),
        vec![1, 2, 3]
    );

    workload
        .add_program_to_full_mesh(&mesh, program)
        .expect("workload should accept program");
    mesh.enqueue_workload(&mut workload, true)
        .expect("enqueue should succeed");

    assert!(mesh.close().expect("mesh close should succeed"));
}

// ── Program: default and runtime_id ──

#[test]
fn program_default_is_same_as_new() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let p1 = Program::new();
    let p2 = Program::default();
    assert_eq!(p1.runtime_id(), p2.runtime_id());
}

#[test]
fn program_runtime_id_wraps_u64_max() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut program = Program::new();
    program.set_runtime_id(u64::MAX);
    assert_eq!(program.runtime_id(), Some(u64::MAX));
    program.set_runtime_id(0);
    assert_eq!(program.runtime_id(), Some(0));
}

// ── MeshWorkload: creation and default ──

#[test]
fn mesh_workload_default_is_same_as_new() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");

    let w1 = MeshWorkload::new();
    let w2 = MeshWorkload::default();
    assert_eq!(w1.program_count(), w2.program_count());
    assert_eq!(w1.program_count(), 0);

    assert!(mesh.close().expect("mesh close should succeed"));
}

#[test]
fn mesh_workload_debug_string() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut mesh = MeshDevice::create_unit_mesh(0).expect("unit mesh should open");

    let workload = MeshWorkload::new();
    assert!(!format!("{workload:?}").is_empty());

    assert!(mesh.close().expect("mesh close should succeed"));
}

// ── Buffer: sub_device_id ──

#[test]
fn interleaved_buffer_sub_device_id_is_none_by_default() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let mut device = Device::create(0).expect("device 0 should open");

    let buffer = Buffer::create_interleaved(
        &device,
        dram_buffer_config(4096),
        BufferCreateOptions::new(),
    )
    .expect("DRAM buffer should allocate");

    assert!(buffer.sub_device_id().is_none());

    drop(buffer);
    assert!(device.close().expect("close should succeed"));
}

// ── CircuarBufferConfig: address_offset ──

#[test]
fn circular_buffer_config_address_offset_reads_back() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();

    let mut cb_config = CircularBufferConfig::new(256);
    cb_config.set_address_offset(128);
    cb_config
        .index(0)
        .set_data_format(DataFormat::RawUInt8)
        .set_page_size(64);

    let cb_id = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect("program should create a circular buffer with address offset");

    let snapshot = program
        .circular_buffer_config(cb_id)
        .expect("circular buffer config should read back");
    assert_eq!(snapshot.address_offset, 128);

    drop(device);
}

// ── CircuarBufferConfig: set_total_size ──

#[test]
fn circular_buffer_config_explicit_total_size_reads_back() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();

    let mut cb_config = CircularBufferConfig::new(128);
    cb_config.set_total_size(384);
    cb_config
        .index(0)
        .set_data_format(DataFormat::RawUInt8)
        .set_page_size(64);

    let cb_id = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect("program should create a circular buffer");

    let snapshot = program
        .circular_buffer_config(cb_id)
        .expect("circular buffer config should read back");
    assert_eq!(snapshot.total_size, 384);

    drop(device);
}

// ── CircuarBufferIndex: set_total_size ──

#[test]
fn circular_buffer_index_total_size_persists() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();

    let mut cb_config = CircularBufferConfig::new(256);
    cb_config
        .index(0)
        .set_data_format(DataFormat::RawUInt8)
        .set_total_size(128)
        .set_page_size(64);

    let cb_id = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect("program should create a circular buffer");

    let snapshot = program
        .circular_buffer_config(cb_id)
        .expect("circular buffer config should read back");
    assert_eq!(
        snapshot.total_size, 128,
        "index-level total_size takes effect"
    );

    drop(device);
}

// ── CircuarBufferIndex: set_tile ──

#[test]
fn circular_buffer_index_set_tile_reads_back() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();

    let mut cb_config = CircularBufferConfig::new(256);
    let tile = TileConfig::new(32, 32, false);
    cb_config
        .index(0)
        .set_data_format(DataFormat::RawUInt8)
        .set_page_size(64)
        .set_tile(tile)
        .expect("set_tile should succeed with device context");

    let cb_id = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect("program should create a circular buffer with tile");

    let snapshot = program
        .circular_buffer_config(cb_id)
        .expect("circular buffer config should read back");
    let index_config = &snapshot.indices[0];
    assert!(index_config.tile.is_some());
    let back = index_config.tile.unwrap();
    assert_eq!(back.height, 32);
    assert_eq!(back.width, 32);
    assert!(!back.transpose_tile);

    drop(device);
}

// ── CircuarBufferConfig: remote_index ──

#[test]
fn circular_buffer_config_can_configure_remote_index() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();

    let backing =
        Buffer::create_interleaved(&device, l1_buffer_config(1024), BufferCreateOptions::new())
            .expect("backing L1 buffer should allocate");

    let mut cb_config = CircularBufferConfig::new(256);
    cb_config
        .set_globally_allocated_address_and_total_size(&backing, 256)
        .expect("global address and total size should set");
    cb_config
        .remote_index(1)
        .set_data_format(DataFormat::Float16)
        .set_page_size(64);

    let error = program
        .create_circular_buffer(&single_core_range_set(), &cb_config)
        .expect_err("remote index without GlobalCircularBuffer should be rejected");
    assert!(
        error.what().contains("Remote buffer indices"),
        "unexpected error: {}",
        error.what()
    );

    drop(device);
}

// ── Empty CoreRangeSet error paths ──

#[test]
fn create_circular_buffer_rejects_empty_core_range_set() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();
    let empty_set = CoreRangeSet::new();
    let cb_config = CircularBufferConfig::new(256);

    let error = program
        .create_circular_buffer(&empty_set, &cb_config)
        .expect_err("empty core range set should be rejected");
    assert!(
        error.what().contains("at least one core range"),
        "unexpected error message: {}",
        error.what()
    );

    drop(device);
}

#[test]
fn create_semaphore_rejects_empty_core_range_set() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let device = Device::create(0).expect("device 0 should open");
    let mut program = Program::new();
    let empty_set = CoreRangeSet::new();

    let error = program
        .create_semaphore(&empty_set, 7)
        .expect_err("empty core range set should be rejected");
    assert!(
        error.what().contains("at least one core range"),
        "unexpected error message: {}",
        error.what()
    );

    drop(device);
}

// ── tilize / untilize round-trip ──

#[test]
fn tilize_untilize_round_trip_bf16_single_tile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let m = 32u32;
    let n = 32u32;
    let elem_size = 2u32;
    let tile_bytes = (m * n * elem_size) as usize;

    let mut input = vec![0u8; tile_bytes];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let tilized = tilize(&input, m, n, elem_size).expect("tilize should succeed");
    assert!(!tilized.is_empty());

    let untilized = untilize(&tilized, m, n, elem_size).expect("untilize should succeed");
    assert_eq!(input, untilized);
}

#[test]
fn tilize_untilize_round_trip_fp32_single_tile() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let m = 32u32;
    let n = 32u32;
    let elem_size = 4u32;
    let tile_bytes = (m * n * elem_size) as usize;

    let mut input = vec![0u8; tile_bytes];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let tilized = tilize(&input, m, n, elem_size).expect("tilize should succeed");
    let untilized = untilize(&tilized, m, n, elem_size).expect("untilize should succeed");
    assert_eq!(input, untilized);
}

#[test]
fn tilize_untilize_round_trip_bf16_four_tiles() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let m = 64u32;
    let n = 64u32;
    let elem_size = 2u32;
    let total_bytes = (m * n * elem_size) as usize;

    let mut input = vec![0u8; total_bytes];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let tilized = tilize(&input, m, n, elem_size).expect("tilize should succeed");
    let untilized = untilize(&tilized, m, n, elem_size).expect("untilize should succeed");
    assert_eq!(input, untilized);
}

#[test]
fn tilize_rejects_non_multiple_of_32() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let data = vec![0u8; 100];
    let result = tilize(&data, 16, 16, 2);
    assert!(result.is_err(), "non-32-multiple dimensions should fail");
}

#[test]
fn tilize_rejects_empty_input() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let result = tilize(&[], 0, 0, 2);
    assert!(
        result.is_err(),
        "empty input should be rejected by TT-Metal"
    );
}

#[test]
fn untilize_rejects_empty_input() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let result = untilize(&[], 0, 0, 2);
    assert!(
        result.is_err(),
        "empty input should be rejected by TT-Metal"
    );
}

#[test]
fn tilize_rejects_elem_size_1() {
    if !hardware_tests_enabled() {
        return;
    }

    let _guard = device_lock();
    let m = 32u32;
    let n = 32u32;
    let elem_size = 1u32;
    let tile_bytes = (m * n * elem_size) as usize;
    let input = vec![0u8; tile_bytes];

    let result = tilize(&input, m, n, elem_size);
    assert!(
        result.is_err(),
        "elem_size=1 should be rejected by TT-Metal"
    );
}
