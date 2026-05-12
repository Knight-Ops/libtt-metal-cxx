use libtt_metal_cxx::{
    Buffer, BufferCreateOptions, BufferType, CoreRangeSet, Device, InterleavedBufferConfig,
    LogicalCore, Program, ShardOrientation, ShardSpecBuffer, ShardedBufferConfig,
    TensorMemoryLayout, available_device_count,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if available_device_count()? == 0 {
        println!("No TT-Metal devices available.");
        return Ok(());
    }

    let device = Device::create(0)?;
    let mut program = Program::new();

    let dram_buffer = Buffer::create_interleaved(
        &device,
        InterleavedBufferConfig::new(4096, 4096, BufferType::Dram),
        BufferCreateOptions::new(),
    )?;
    println!(
        "DRAM buffer: address={}, size={}, allocated={}",
        dram_buffer.address(),
        dram_buffer.size(),
        dram_buffer.is_allocated()
    );

    program.assign_global_buffer(&dram_buffer)?;
    println!("Assigned DRAM buffer to program.");

    let sharded_spec = ShardSpecBuffer::new(
        CoreRangeSet::from_core(LogicalCore::new(0, 0)),
        [1, 1],
        ShardOrientation::RowMajor,
        [1, 1],
        [1, 1],
    );
    let sharded_buffer = Buffer::create_sharded(
        &device,
        &ShardedBufferConfig::new(
            2048,
            2048,
            BufferType::L1,
            TensorMemoryLayout::HeightSharded,
            sharded_spec,
        ),
        BufferCreateOptions::new(),
    )?;
    let shard_spec = sharded_buffer
        .shard_spec()?
        .expect("sharded buffer should expose shard metadata");
    println!(
        "Sharded buffer: layout={:?}, shards={}",
        sharded_buffer.buffer_layout(),
        shard_spec.grid.ranges().len()
    );

    let mut scratch_l1 = Buffer::create_interleaved(
        &device,
        InterleavedBufferConfig::new(2048, 2048, BufferType::L1),
        BufferCreateOptions::new(),
    )?;
    println!(
        "Scratch L1 buffer before deallocate: allocated={}",
        scratch_l1.is_allocated()
    );
    println!("Scratch L1 deallocated: {}", scratch_l1.deallocate()?);
    println!(
        "Scratch L1 buffer after deallocate: allocated={}",
        scratch_l1.is_allocated()
    );

    Ok(())
}
