use std::env;
use std::path::{Path, PathBuf};

fn env_path(var: &str) -> Option<PathBuf> {
    env::var_os(var).map(PathBuf::from)
}

fn require_path(path: &Path, kind: &str) {
    if !path.exists() {
        panic!("required {kind} not found at {}", path.display());
    }
}

fn is_runtime_root(path: &Path) -> bool {
    path.join("tt_metal").is_dir()
}

fn main() {
    let install_dir =
        env_path("TT_METAL_INSTALL_DIR").unwrap_or_else(|| PathBuf::from("/usr/local"));
    let include_dir =
        env_path("TT_METAL_INCLUDE_DIR").unwrap_or_else(|| install_dir.join("include"));
    let lib_dir = env_path("TT_METAL_LIB_DIR").unwrap_or_else(|| install_dir.join("lib"));
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let workspace_tt_metal = manifest_dir
        .parent()
        .map(|parent| parent.join("tt-metal"))
        .filter(|path| is_runtime_root(path));
    let runtime_root = env_path("TT_METAL_RUNTIME_ROOT")
        .filter(|path| is_runtime_root(path))
        .or_else(|| env_path("TT_METAL_HOME").filter(|path| is_runtime_root(path)))
        .or_else(|| {
            let candidate = install_dir.join("libexec/tt-metalium");
            is_runtime_root(&candidate).then_some(candidate)
        })
        .or_else(|| {
            let candidate = install_dir.join("share/tt-metalium/tests");
            is_runtime_root(&candidate).then_some(candidate)
        })
        .or(workspace_tt_metal);

    require_path(
        &include_dir.join("tt-metalium/host_api.hpp"),
        "TT-Metal host API header",
    );
    require_path(&lib_dir.join("libtt_metal.so"), "TT-Metal shared library");

    let mut build = cxx_build::bridge("src/ffi.rs");
    build
        .file("src/tt_metal_cxx/runtime.cc")
        .file("src/tt_metal_cxx/device.cc")
        .file("src/tt_metal_cxx/kernel.cc")
        .file("src/tt_metal_cxx/program.cc")
        .file("src/tt_metal_cxx/runtime_args.cc")
        .file("src/tt_metal_cxx/distributed.cc")
        .include(".")
        .include("include")
        .include(&include_dir)
        .define("SPDLOG_FMT_EXTERNAL", "1")
        .std("c++20");

    let runtime_root_define = runtime_root
        .as_ref()
        .map(|path| format!("\"{}\"", path.display()));
    if let Some(runtime_root_define) = runtime_root_define.as_deref() {
        build.define("TT_METAL_DEFAULT_RUNTIME_ROOT", runtime_root_define);
    }

    build.compile("tt-metal-cxx");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=fmt");
    println!("cargo:rustc-link-lib=dylib=tt_stl");
    println!("cargo:rustc-link-lib=dylib=tt_metal");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!(
        "cargo:rustc-link-arg-tests=-Wl,-rpath,{}",
        lib_dir.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx.hpp");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/device.rs");
    println!("cargo:rerun-if-changed=src/distributed.rs");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/kernel.rs");
    println!("cargo:rerun-if-changed=src/program.rs");
    println!("cargo:rerun-if-changed=src/runtime_args.rs");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/runtime.cc");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/device.cc");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/kernel.cc");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/program.cc");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/runtime_args.cc");
    println!("cargo:rerun-if-changed=src/tt_metal_cxx/distributed.cc");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx/runtime.hpp");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx/device.hpp");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx/kernel.hpp");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx/program.hpp");
    println!("cargo:rerun-if-changed=include/tt_metal_cxx/distributed.hpp");
    println!("cargo:rerun-if-env-changed=TT_METAL_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=TT_METAL_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TT_METAL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TT_METAL_HOME");
    println!("cargo:rerun-if-env-changed=TT_METAL_RUNTIME_ROOT");
}
