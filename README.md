# libtt-metal-cxx

`libtt-metal-cxx` is a small Rust crate that exposes a focused TT-Metal host API through a C++ bridge built with [`cxx`](https://cxx.rs/).

It currently wraps:

- device discovery
- device open/close lifecycle
- basic `Program` creation and `runtime_id` access

## Layout

- `src/lib.rs`: Rust API and `cxx::bridge` declarations
- `include/tt_metal_cxx.hpp`: public C++ bridge types and function declarations
- `src/tt_metal_cxx.cc`: TT-Metal-backed C++ implementation
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
```

## Rust API

```rust
use libtt_metal_cxx::{query_devices, Device, Program};

let counts = query_devices()?;
let mut device = Device::create(0)?;
let mut program = Program::create();

program.set_runtime_id(42);
assert_eq!(program.runtime_id(), Some(42));

device.close()?;
# Ok::<(), libtt_metal_cxx::Exception>(())
```

Main entry points:

- `query_devices()`, `available_device_count()`, `pcie_device_count()`
- `Device::create()`, `Device::close()`, `Device::is_open()`, `Device::device_id()`
- `Program::create()`, `Program::runtime_id()`, `Program::set_runtime_id()`
