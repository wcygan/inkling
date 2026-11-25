//! Raw CUDA driver API imported straight from `nvcuda.dll` (no CUDA Toolkit needed at run time).
//! Signatures follow `cuda.h` from CUDA 13.5; the `_v2` names are the ones the header's macros select.

#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms, dead_code)]

use std::ffi::{c_char, c_void};

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUgraphConditionalHandle = u64;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: i32 = 1;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: i32 = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: i32 = 4;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
pub const CU_STREAM_NON_BLOCKING: u32 = 1;
pub const CU_EVENT_DEFAULT: u32 = 0;
pub const CU_EVENT_DISABLE_TIMING: u32 = 2;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: i32 = 6;
pub const CU_GRAPH_NODE_TYPE_CONDITIONAL: i32 = 13;
pub const CU_GRAPH_COND_TYPE_SWITCH: i32 = 2;
pub const CU_STREAM_SET_CAPTURE_DEPENDENCIES: u32 = 1;

/// `CUgraphNodeParams` (256 bytes) with the `CUDA_CONDITIONAL_NODE_PARAMS` member of its union, which
/// starts at byte 16: handle, type, size, `phGraph_out` (CUDA-owned array, set by `cuGraphAddNode`), ctx.
#[repr(C)]
pub struct CUgraphNodeParamsConditional {
    pub node_type: i32,
    pub reserved0: [i32; 3],
    pub handle: CUgraphConditionalHandle,
    pub cond_type: i32,
    pub size: u32,
    pub ph_graph_out: *mut CUgraph,
    pub ctx: CUcontext,
    pub pad: [u8; 232 - 32],
    pub reserved2: i64,
}
const _: () = assert!(std::mem::size_of::<CUgraphNodeParamsConditional>() == 256);

/// `CUlaunchAttribute`: a 4-byte id, padding to 8, then a 64-byte value union.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUlaunchAttribute {
    pub id: i32,
    pub pad: [u8; 4],
    pub value: [u8; 64],
}

/// `CUlaunchConfig`.
#[repr(C)]
pub struct CUlaunchConfig {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: u32,
    pub h_stream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub num_attrs: u32,
}

#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "system" {
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDriverGetVersion(v: *mut i32) -> CUresult;
    pub fn cuDeviceGet(d: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: i32, d: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(v: *mut i32, attrib: i32, d: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, d: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(d: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuGetErrorName(r: CUresult, s: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(r: CUresult, s: *mut *const c_char) -> CUresult;

    pub fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(m: CUmodule) -> CUresult;
    pub fn cuModuleGetFunctionCount(count: *mut u32, m: CUmodule) -> CUresult;
    pub fn cuModuleEnumerateFunctions(fns: *mut CUfunction, n: u32, m: CUmodule) -> CUresult;
    pub fn cuFuncGetName(name: *mut *const c_char, f: CUfunction) -> CUresult;
    pub fn cuFuncGetAttribute(v: *mut i32, attrib: i32, f: CUfunction) -> CUresult;
    pub fn cuFuncSetAttribute(f: CUfunction, attrib: i32, v: i32) -> CUresult;

    pub fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemHostAlloc(p: *mut *mut c_void, bytes: usize, flags: u32) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, s: CUstream) -> CUresult;
    pub fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: CUstream) -> CUresult;
    pub fn cuMemsetD32Async(dst: CUdeviceptr, v: u32, n: usize, s: CUstream) -> CUresult;

    pub fn cuStreamCreate(s: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamDestroy_v2(s: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(s: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(s: CUstream, e: CUevent, flags: u32) -> CUresult;
    pub fn cuEventCreate(e: *mut CUevent, flags: u32) -> CUresult;
    pub fn cuEventDestroy_v2(e: CUevent) -> CUresult;
    pub fn cuEventRecord(e: CUevent, s: CUstream) -> CUresult;
    pub fn cuEventSynchronize(e: CUevent) -> CUresult;
    pub fn cuEventElapsedTime_v2(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    pub fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    pub fn cuStreamBeginCapture_v2(s: CUstream, mode: i32) -> CUresult;
    pub fn cuStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> CUresult;
    pub fn cuGraphUpload(e: CUgraphExec, s: CUstream) -> CUresult;
    pub fn cuGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(e: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(g: CUgraph) -> CUresult;
    pub fn cuGraphGetNodes(g: CUgraph, nodes: *mut *mut c_void, n: *mut usize) -> CUresult;

    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    pub fn cuGraphCreate(g: *mut CUgraph, flags: u32) -> CUresult;
    pub fn cuGraphConditionalHandleCreate(
        h: *mut CUgraphConditionalHandle,
        g: CUgraph,
        ctx: CUcontext,
        default_value: u32,
        flags: u32,
    ) -> CUresult;
    pub fn cuStreamBeginCaptureToGraph(
        s: CUstream,
        g: CUgraph,
        deps: *const CUgraphNode,
        edge_data: *const c_void,
        n: usize,
        mode: i32,
    ) -> CUresult;
    pub fn cuStreamGetCaptureInfo_v3(
        s: CUstream,
        status: *mut i32,
        id: *mut u64,
        g: *mut CUgraph,
        deps: *mut *const CUgraphNode,
        edge_data: *mut *const c_void,
        n: *mut usize,
    ) -> CUresult;
    pub fn cuStreamUpdateCaptureDependencies_v2(
        s: CUstream,
        deps: *mut CUgraphNode,
        edge_data: *const c_void,
        n: usize,
        flags: u32,
    ) -> CUresult;
    pub fn cuGraphAddNode_v2(
        node: *mut CUgraphNode,
        g: CUgraph,
        deps: *const CUgraphNode,
        edge_data: *const c_void,
        n: usize,
        params: *mut CUgraphNodeParamsConditional,
    ) -> CUresult;
}
