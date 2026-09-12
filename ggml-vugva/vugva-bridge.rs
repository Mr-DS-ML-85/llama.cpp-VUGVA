//! C FFI bridge for llama.cpp integration.
//!
//! Exposes the VUGVA tiered memory pool as a C API so llama.cpp (or any C/C++
//! consumer) can allocate DRAM, promote to VRAM, and demote back — without
//! touching the Rust types directly.
//!
//! All functions are `extern "C"` and use opaque handles. No Rust types leak
//! into the C header.

use crate::tiered::TieredPool;
use crate::vmt::Tier;
use std::ffi::CStr;

/// Opaque handle wrapping a VUGVA tiered pool.
#[repr(C)]
pub struct VugvaPool {
    inner: TieredPool,
}

/// Pool statistics returned by `vugva_pool_info`.
#[repr(C)]
pub struct VugvaPoolInfo {
    pub dram_capacity: usize,
    pub dram_used: usize,
    pub numa_bound: std::ffi::c_int,
    pub pinned: std::ffi::c_int,
}

/// Create a tiered pool with `dram_bytes` of DRAM on GPU `gpu_id`.
///
/// Returns null on failure. Call `vugva_pool_destroy` to release.
#[no_mangle]
pub extern "C" fn vugva_pool_create(gpu_id: i32, dram_bytes: usize) -> *mut VugvaPool {
    match TieredPool::new(&[gpu_id], dram_bytes) {
        Ok(inner) => Box::into_raw(Box::new(VugvaPool { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Destroy a pool previously created by `vugva_pool_create`.
///
/// # Safety
///
/// `pool` must be a pointer returned by `vugva_pool_create` and not used after.
#[no_mangle]
pub unsafe extern "C" fn vugva_pool_destroy(pool: *mut VugvaPool) {
    if !pool.is_null() {
        unsafe { drop(Box::from_raw(pool)); }
    }
}

/// Allocate `total_bytes` in the pool as a named page, starting in DRAM.
///
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn vugva_alloc(
    pool: *mut VugvaPool,
    name: *const std::ffi::c_char,
    total_bytes: usize,
) -> i32 {
    if pool.is_null() || name.is_null() {
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let shape = [total_bytes];
    match unsafe { &mut (*pool).inner }.allocate(name_str, &shape, 1, Tier::Dram) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Copy `data[0..len]` into a named page's DRAM backing.
///
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn vugva_write(
    pool: *mut VugvaPool,
    name: *const std::ffi::c_char,
    data: *const u8,
    len: usize,
) -> i32 {
    if pool.is_null() || name.is_null() || data.is_null() {
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    match unsafe { &mut (*pool).inner }.write_page(name_str, slice) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Promote `name` to VRAM on GPU 0, returning its device pointer.
///
/// Returns 0 on success (device pointer written to `*out_ptr`), -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn vugva_access(
    pool: *mut VugvaPool,
    name: *const std::ffi::c_char,
    out_ptr: *mut u64,
) -> i32 {
    if pool.is_null() || name.is_null() || out_ptr.is_null() {
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match unsafe { &mut (*pool).inner }.access(name_str, 0) {
        Ok(ptr) => {
            unsafe { *out_ptr = ptr; }
            0
        }
        Err(_) => -1,
    }
}

/// Demote `name` from VRAM back to DRAM.
///
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn vugva_demote(
    pool: *mut VugvaPool,
    name: *const std::ffi::c_char,
) -> i32 {
    if pool.is_null() || name.is_null() {
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match unsafe { &mut (*pool).inner }.demote(name_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Deallocate `name` entirely, returning DRAM and VRAM to their pools.
///
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn vugva_dealloc(
    pool: *mut VugvaPool,
    name: *const std::ffi::c_char,
) -> i32 {
    if pool.is_null() || name.is_null() {
        return -1;
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match unsafe { &mut (*pool).inner }.deallocate(name_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get pool statistics.
#[no_mangle]
pub unsafe extern "C" fn vugva_pool_info(
    pool: *const VugvaPool,
    out: *mut VugvaPoolInfo,
) -> i32 {
    if pool.is_null() || out.is_null() {
        return -1;
    }
    let p = unsafe { &*pool };
    let info = VugvaPoolInfo {
        dram_capacity: p.inner.dram_capacity(0).unwrap_or(0),
        dram_used: p.inner.dram_used(0).unwrap_or(0),
        numa_bound: p.inner.dram_numa_bound(0).unwrap_or(false) as std::ffi::c_int,
        pinned: p.inner.dram_pinned(0).unwrap_or(false) as std::ffi::c_int,
    };
    unsafe { *out = info; }
    0
}
