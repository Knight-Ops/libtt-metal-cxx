# libtt-metal-cxx

`libtt-metal-cxx` is a small Rust crate that exposes a focused TT-Metal host API through a C++ bridge built with [`cxx`](https://cxx.rs/).

## Features

- [x] Device management: `CreateDevice`, `CloseDevice`, and `QueryDevices`
- [x] Program creation: `CreateProgram`
- [x] Program metadata: `runtime_id` get/set
- [x] Kernel creation from source files: `CreateKernel` for compute and data-movement kernels
- [x] Kernel creation from inline source: `CreateKernelFromString` for compute and data-movement kernels
- [x] Compute kernel configuration builders
- [x] Data-movement kernel configuration builders
- [x] Runtime arguments: per-core set/get
- [x] Runtime arguments: common set/get
- [x] Buffer creation: interleaved `CreateBuffer`
- [x] Buffer creation: sharded `CreateBuffer`
- [x] Buffer lifetime helpers: `DeallocateBuffer`
- [x] Program-owned global buffers: `AssignGlobalBufferToProgram`
- [x] Circular buffer creation: `CreateCircularBuffer`
- [x] Circular buffer readback: `GetCircularBufferConfig`
- [x] Circular buffer updates: `UpdateCircularBufferTotalSize`
- [x] Circular buffer updates: `UpdateCircularBufferPageSize`
- [x] Circular buffer updates: `UpdateDynamicCircularBufferAddress`
- [x] Circular buffer updates: `UpdateDynamicCircularBufferAddress(..., address_offset)`
- [x] Circular buffer updates: `UpdateDynamicCircularBufferAddressAndTotalSize`
- [x] Semaphore creation: `CreateSemaphore`
- [x] Wrapped distributed workflow: `MeshDevice::create_unit_mesh`, `MeshWorkload`, and mesh enqueue
- [x] Runnable examples for device, program, kernels, buffers, and program resources
- [x] Hardware-backed integration tests for the implemented bindings
- [ ] Profiler APIs: `ReadMeshDeviceProfilerResults`, `GetLatestProgramsPerfData`, `GetAllProgramsPerfData`
- [ ] Full kernel placement parity for `CoreRange` and `CoreRangeSet`
- [ ] Ethernet kernel support / `EthernetConfig`
- [ ] Full runtime-args parity for bulk multi-core setters/getters
- [ ] Full runtime-args parity for whole-kernel readback helpers
- [ ] Global circular buffer workflows / remote circular-buffer support on top of the installed TT-Metal APIs
- [ ] Richer distributed APIs: mesh events, tracing, and more mesh-side execution helpers
- [ ] Mesh/distributed buffer workflows beyond the current wrapped unit-mesh path
- [ ] Sub-device manager APIs and stronger sub-device hardware coverage

## Layout

- `src/lib.rs`: Rust API re-exports
- `src/ffi.rs`: `cxx::bridge` declarations
- `src/{device,program,kernel,runtime_args,distributed,buffer}.rs`: Rust subsystem APIs
- `include/tt_metal_cxx.hpp`: public C++ bridge types and function declarations
- `include/tt_metal_cxx/*.hpp`: C++ subsystem bridge headers
- `src/tt_metal_cxx/*.cc`: TT-Metal-backed C++ implementation
- `build.rs`: TT-Metal path resolution, C++ build, and linker setup
- `examples/`: small runnable examples
- `tests/`: integration tests, including hardware-backed coverage

## Requirements

You need a TT-Metal installation that provides:

- `tt-metalium/host_api.hpp`
- `libtt_metal.so`
- a runtime root containing `tt_metal/`

By default the build looks under `/usr/local`, but you can override paths with:

- `TT_METAL_INSTALL_DIR`
- `TT_METAL_INCLUDE_DIR`
- `TT_METAL_LIB_DIR`
- `TT_METAL_RUNTIME_ROOT`
- `TT_METAL_HOME`

`build.rs` also tries common runtime-root locations under the install dir and a sibling `../tt-metal` workspace checkout.

## Build

```bash
cargo build
```

## Test

Hardware-backed tests are gated behind `TT_METAL_RUN_HARDWARE_TESTS=1`.

```bash
cargo test
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test
```

## Examples

```bash
cargo run --example create_program
cargo run --example create_close
cargo run --example create_kernels
cargo run --example create_kernels_from_string
cargo run --example runtime_args
cargo run --example unit_mesh_workload
cargo run --example buffer_lifecycle
cargo run --example program_resources
```

## Rust API

```rust
use libtt_metal_cxx::{
    query_devices, Buffer, BufferCreateOptions, BufferType, Device, InterleavedBufferConfig,
    Program,
};

let counts = query_devices()?;
let mut device = Device::create(0)?;
let mut program = Program::create();
let mut buffer = Buffer::create_interleaved(
    &device,
    InterleavedBufferConfig::new(4096, 4096, BufferType::Dram),
    BufferCreateOptions::new(),
)?;

program.set_runtime_id(42);
assert_eq!(program.runtime_id(), Some(42));
program.assign_global_buffer(&buffer)?;
buffer.deallocate()?;

device.close()?;
# Ok::<(), libtt_metal_cxx::Exception>(())
```

Main entry points:

- `query_devices()`, `available_device_count()`, `pcie_device_count()`
- `Device::create()`, `Device::close()`, `Device::is_open()`, `Device::device_id()`
- `Program::create()`, `Program::runtime_id()`, `Program::set_runtime_id()`
- `Program::{create_compute_kernel*, create_data_movement_kernel*}`
- `Program::{set_runtime_args, runtime_args, set_common_runtime_args, common_runtime_args}`
- `Buffer::{create_interleaved, create_sharded, deallocate}`
- `Program::{assign_global_buffer, create_circular_buffer, circular_buffer_config, create_semaphore}`
- `MeshDevice::create_unit_mesh()`, `MeshWorkload::create()`, `MeshDevice::enqueue_workload()`
