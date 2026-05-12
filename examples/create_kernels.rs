use libtt_metal_cxx::{
    ComputeKernelConfig, DataMovementKernelConfig, DataMovementProcessor, KernelBuildOptLevel,
    LogicalCore, MathFidelity, MeshDevice, MeshWorkload, Program, query_devices,
};

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
        .add_compile_args([11, 22, 33])
        .add_define("EXAMPLE_COMPUTE_DEFINE", "1")
        .add_named_compile_arg("example_compute_arg", 7);
    let mut dataflow_config = DataMovementKernelConfig::reader()?;
    dataflow_config
        .set_processor(DataMovementProcessor::Riscv1)
        .add_compile_args([5, 6])
        .add_define("EXAMPLE_DM_DEFINE", "1")
        .add_named_compile_arg("example_dm_arg", 9)
        .set_opt_level(KernelBuildOptLevel::O2);

    let compute_kernel_id = program.create_compute_kernel_with_config(
        "tt_metal/kernels/compute/blank.cpp",
        core,
        &compute_config,
    )?;
    let dataflow_kernel_id = program.create_data_movement_kernel_with_config(
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        &dataflow_config,
    )?;

    println!(
        "Created kernels: compute={}, dataflow={}",
        compute_kernel_id, dataflow_kernel_id
    );

    workload.add_program_to_full_mesh(&mesh, program)?;
    mesh.enqueue_workload(&mut workload, true)?;
    println!("Enqueued kernel workload successfully");

    let closed = mesh.close()?;
    println!("Closed mesh 0: {closed}");

    Ok(())
}
