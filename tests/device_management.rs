use std::env;
use std::sync::{Mutex, MutexGuard, OnceLock};

use libtt_metal_cxx::{
    ComputeKernelConfig, DataMovementKernelConfig, DataMovementProcessor, Device,
    KernelBuildOptLevel, LogicalCore, MathFidelity, MeshDevice, MeshWorkload, Program,
    available_device_count, pcie_device_count, query_devices,
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
