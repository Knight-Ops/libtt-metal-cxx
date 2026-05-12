use libtt_metal_cxx::{
    Buffer, BufferCreateOptions, BufferType, CircularBufferConfig, CoreRange, CoreRangeSet,
    DataFormat, Device, InterleavedBufferConfig, LogicalCore, Program, available_device_count,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if available_device_count()? == 0 {
        println!("No TT-Metal devices available.");
        return Ok(());
    }

    let device = Device::create(0)?;
    let mut program = Program::new();
    let core0 = LogicalCore::new(0, 0);
    let core1 = LogicalCore::new(0, 1);
    let single_core = CoreRangeSet::from_core(core0);
    let two_ranges =
        CoreRangeSet::from_ranges([CoreRange::from_core(core0), CoreRange::from_core(core1)]);

    let backing = Buffer::create_interleaved(
        &device,
        InterleavedBufferConfig::new(1024, 1024, BufferType::L1),
        BufferCreateOptions::new(),
    )?;
    let replacement = Buffer::create_interleaved(
        &device,
        InterleavedBufferConfig::new(1024, 1024, BufferType::L1),
        BufferCreateOptions::new(),
    )?;

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
    cb_config.set_globally_allocated_address(&backing)?;

    let cb_id = program.create_circular_buffer(&single_core, &cb_config)?;
    let initial = program.circular_buffer_config(cb_id)?;
    println!(
        "Circular buffer {}: total_size={}, global_addr={:?}, indices={}",
        cb_id,
        initial.total_size,
        initial.globally_allocated_address,
        initial.indices.len()
    );

    let sem_single = program.create_semaphore(&single_core, 7)?;
    let sem_multi = program.create_semaphore(&two_ranges, 11)?;
    println!("Semaphores: single={}, multi={}", sem_single, sem_multi);

    program.update_circular_buffer_total_size(cb_id, 512)?;
    program.update_circular_buffer_page_size(cb_id, 0, 128)?;
    program.update_dynamic_circular_buffer_address(cb_id, &replacement)?;
    program.update_dynamic_circular_buffer_address_with_offset(cb_id, &replacement, 32)?;
    program.update_dynamic_circular_buffer_address_and_total_size(cb_id, &replacement, 512)?;

    let updated = program.circular_buffer_config(cb_id)?;
    println!(
        "Updated circular buffer: total_size={}, global_addr={:?}, offset={}",
        updated.total_size, updated.globally_allocated_address, updated.address_offset
    );

    Ok(())
}
