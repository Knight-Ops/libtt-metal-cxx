use libtt_metal_cxx::{
    ComputeKernelConfig, DataMovementKernelConfig, DataMovementProcessor, KernelBuildOptLevel,
    LogicalCore, MathFidelity, MeshDevice, MeshWorkload, Program, query_devices,
};

const INLINE_COMPUTE_KERNEL_SOURCE: &str = r#"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {}
"#;

const INLINE_DATAFLOW_KERNEL_SOURCE: &str = r#"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let counts = query_devices()?;
    println!(
        "TT-Metal devices: available={}, pcie={}",
        counts.available, counts.pcie
    );

    if counts.available == 0 {
        println!("No TT-Metal devices are available to open.");
        return Ok(());
    }

    let mut mesh = MeshDevice::create_unit_mesh(0)?;
    let mut workload = MeshWorkload::new();
    let mut program = Program::new();
    let core = LogicalCore::new(0, 0);
    let mut compute_config = ComputeKernelConfig::new();
    compute_config
        .set_math_fidelity(MathFidelity::HiFi4)
        .set_opt_level(KernelBuildOptLevel::O3)
        .add_compile_arg(17);
    let mut dataflow_config = DataMovementKernelConfig::reader()?;
    dataflow_config
        .set_processor(DataMovementProcessor::Riscv1)
        .add_compile_arg(23)
        .set_opt_level(KernelBuildOptLevel::O2);

    let compute_kernel_id = program.create_compute_kernel_from_string_with_config(
        INLINE_COMPUTE_KERNEL_SOURCE,
        core,
        &compute_config,
    )?;
    let dataflow_kernel_id = program.create_data_movement_kernel_from_string_with_config(
        INLINE_DATAFLOW_KERNEL_SOURCE,
        core,
        &dataflow_config,
    )?;

    println!(
        "Created inline kernels: compute={}, dataflow={}",
        compute_kernel_id, dataflow_kernel_id
    );

    workload.add_program_to_full_mesh(&mesh, program)?;
    mesh.enqueue_workload(&mut workload, true)?;
    println!("Enqueued inline kernel workload successfully");

    let closed = mesh.close()?;
    println!("Closed mesh 0: {closed}");

    Ok(())
}
