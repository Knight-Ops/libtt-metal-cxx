use libtt_metal_cxx::{
    DataMovementProcessor, LogicalCore, MeshDevice, MeshWorkload, Noc, Program, query_devices,
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
    let mut workload = MeshWorkload::create();
    let mut program = Program::create();
    let core = LogicalCore::new(0, 0);
    let kernel_id = program.create_data_movement_kernel(
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        DataMovementProcessor::Riscv1,
        Noc::Riscv1Default,
    )?;

    program.set_runtime_args(kernel_id, core, &[11, 22, 33])?;
    program.set_common_runtime_args(kernel_id, &[100, 200])?;

    println!(
        "runtime_args={:?}, common_runtime_args={:?}",
        program.runtime_args(kernel_id, core)?,
        program.common_runtime_args(kernel_id)?,
    );

    workload.add_program_to_full_mesh(&mesh, program)?;
    mesh.enqueue_workload(&mut workload, true)?;
    println!("Enqueued runtime-args workload successfully");

    let closed = mesh.close()?;
    println!("Closed mesh 0: {closed}");

    Ok(())
}
