use libtt_metal_cxx::{MeshDevice, MeshWorkload, Program, query_devices};

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
    println!(
        "Opened unit mesh {:?}: rows={}, cols={}, devices={}",
        mesh.device_id(),
        mesh.num_rows()?,
        mesh.num_cols()?,
        mesh.num_devices()?
    );

    let mut workload = MeshWorkload::new();
    workload.add_program_to_full_mesh(&mesh, Program::new())?;
    println!("Workload program_count={}", workload.program_count());
    mesh.enqueue_workload(&mut workload, true)?;
    println!("Enqueued workload successfully");

    let closed = mesh.close()?;
    println!("Closed mesh 0: {closed}");

    Ok(())
}
