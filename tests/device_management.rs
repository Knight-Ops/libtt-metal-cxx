use std::env;
use std::sync::{Mutex, MutexGuard, OnceLock};

use libtt_metal_cxx::{Device, Program, available_device_count, pcie_device_count, query_devices};

fn device_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
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
