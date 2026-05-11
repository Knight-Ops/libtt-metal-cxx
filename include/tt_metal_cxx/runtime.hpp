#pragma once

#include "rust/cxx.h"

namespace tt_metal_cxx::detail {

void ensure_runtime_root();
void register_atexit_cleanup();
void maybe_release_ownership();
void note_device_opened();
void note_device_closed();
void note_program_opened();
void note_program_closed();
void note_mesh_opened();
void note_mesh_closed();

}  // namespace tt_metal_cxx::detail

namespace tt_metal_cxx {

void throw_invalid_argument(rust::Str message);

}  // namespace tt_metal_cxx
