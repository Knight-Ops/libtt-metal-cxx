#include "tt_metal_cxx/runtime.hpp"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

#include <tt-metalium/host_api.hpp>

namespace tt::tt_metal::detail {
void ReleaseOwnership();
}

namespace tt_metal_cxx::detail {

namespace {

bool is_runtime_root(const std::filesystem::path& candidate) {
    return !candidate.empty() && std::filesystem::is_directory(candidate / "tt_metal");
}

bool set_root_dir_if_valid(std::string_view candidate) {
    if (candidate.empty()) {
        return false;
    }

    if (is_runtime_root(std::filesystem::path(candidate))) {
        tt::tt_metal::SetRootDir(std::string(candidate));
        return true;
    }

    return false;
}

std::mutex& ownership_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::size_t& open_device_count() {
    static std::size_t count = 0;
    return count;
}

std::size_t& open_program_count() {
    static std::size_t count = 0;
    return count;
}

std::size_t& open_mesh_count() {
    static std::size_t count = 0;
    return count;
}

}  // namespace

void ensure_runtime_root() {
    static std::once_flag once;

    std::call_once(once, [] {
        if (const char* runtime_root = std::getenv("TT_METAL_RUNTIME_ROOT")) {
            if (set_root_dir_if_valid(runtime_root)) {
                return;
            }
            throw std::runtime_error(
                "TT_METAL_RUNTIME_ROOT must point to a directory containing tt_metal/");
        }

        if (const char* metal_home = std::getenv("TT_METAL_HOME")) {
            if (set_root_dir_if_valid(metal_home)) {
                return;
            }
            throw std::runtime_error("TT_METAL_HOME must point to a directory containing tt_metal/");
        }

#ifdef TT_METAL_DEFAULT_RUNTIME_ROOT
        if (set_root_dir_if_valid(TT_METAL_DEFAULT_RUNTIME_ROOT)) {
            return;
        }
#endif

        throw std::runtime_error(
            "unable to resolve TT-Metal runtime root; set TT_METAL_RUNTIME_ROOT to a directory containing tt_metal/");
    });
}

void register_atexit_cleanup() {
    static std::once_flag once;

    std::call_once(once, [] {
        std::atexit([] {
            try {
                tt::tt_metal::detail::ReleaseOwnership();
            } catch (...) {
            }
        });
    });
}

void maybe_release_ownership() {
    bool should_release = false;
    {
        std::scoped_lock lock(ownership_mutex());
        should_release = open_device_count() == 0 && open_program_count() == 0 && open_mesh_count() == 0;
    }

    if (should_release) {
        tt::tt_metal::detail::ReleaseOwnership();
    }
}

void note_device_opened() {
    std::scoped_lock lock(ownership_mutex());
    ++open_device_count();
}

void note_device_closed() {
    std::scoped_lock lock(ownership_mutex());
    if (open_device_count() > 0) {
        --open_device_count();
    }
}

void note_program_opened() {
    std::scoped_lock lock(ownership_mutex());
    ++open_program_count();
}

void note_program_closed() {
    std::scoped_lock lock(ownership_mutex());
    if (open_program_count() > 0) {
        --open_program_count();
    }
}

void note_mesh_opened() {
    std::scoped_lock lock(ownership_mutex());
    ++open_mesh_count();
}

void note_mesh_closed() {
    std::scoped_lock lock(ownership_mutex());
    if (open_mesh_count() > 0) {
        --open_mesh_count();
    }
}

}  // namespace tt_metal_cxx::detail
