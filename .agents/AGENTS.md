# libtt-metal-cxx — agent guide

## Build & test

```bash
cargo build
cargo test                           # skips hardware-backed tests
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test   # runs all tests
cargo run --example <name>           # see examples/ for names
```

No CI, no formatter config, no clippy config. `edition = "2024"`.

## TT-Metal dependency

Requires `tt-metalium/host_api.hpp`, `libtt_metal.so`, and a runtime root (`tt_metal/` subdirectory). The crate searches these env vars (in order):
`TT_METAL_INSTALL_DIR` → `TT_METAL_INCLUDE_DIR` / `TT_METAL_LIB_DIR` / `TT_METAL_RUNTIME_ROOT` / `TT_METAL_HOME`. Falls back to `/usr/local` default and a sibling `../tt-metal` workspace checkout. See `build.rs` for the full probe order.

C++ side: `include/tt_metal_cxx*.hpp` (public C++ API), `src/tt_metal_cxx/*.cc` (implementations). Both compiled via `cxx_build` in `build.rs`.

## Architecture

- `src/ffi.rs` — single `cxx::bridge` declaring all FFI signatures and opaque handle types
- `src/{device,program,kernel,buffer,runtime_args,distributed}.rs` — Rust wrappers around the FFI
- `src/lib.rs` — re-exports the public API
- `tests/` — integration tests in a single crate (tests share helper functions across files)
- `generated/` — runtime output from TT-Metal kernel compilation (gitignored, created at runtime)

## Hardware test quirks

- Gated on `TT_METAL_RUN_HARDWARE_TESTS=1` env var being set (any non-empty value)
- All hardware tests acquire a global `Mutex<()>` via `device_lock()` to serialize TT-Metal device access
- Tests early-return (not `#[ignore]`) when hardware is unavailable; no device tests fail spuriously
- Kernel paths (`tt_metal/kernels/compute/blank.cpp`, `tt_metal/kernels/dataflow/blank.cpp`) are relative to the TT-Metal runtime root

## Conventions

- `build.rs` explicitly lists every file it `rerun-if-changed` — adding a new source file requires updating this list
- Adding new FFI functions requires: declaring in `ffi.rs`, implementing C++ side in both `include/` and `src/tt_metal_cxx/`, wrapping in a Rust module, re-exporting from `lib.rs`
