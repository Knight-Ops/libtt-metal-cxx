use libtt_metal_cxx::{Device, query_devices};

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

    let mut device = Device::create(0)?;
    println!(
        "Opened device {:?}, is_open={}",
        device.device_id(),
        device.is_open()
    );

    let closed = device.close()?;
    println!("Closed device 0: {closed}");
    println!("Device open after close: {}", device.is_open());

    Ok(())
}
