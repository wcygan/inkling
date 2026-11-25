use crate::error::{Result, ck, check};
use crate::launch::LaunchConfig;
use crate::stream::Stream;
use crate::sys;
use std::ffi::{CStr, c_char, c_void};

/// A loaded cubin with its kernels indexed by (possibly mangled) name.
pub struct Module {
    raw: sys::CUmodule,
    label: String,
    functions: Vec<(String, sys::CUfunction)>,
}

impl Module {
    /// Loads a cubin image. `label` names the module in errors.
    pub fn from_cubin(image: &[u8], label: impl Into<String>) -> Result<Module> {
        let label = label.into();
        // cuModuleLoadData requires the image to stay valid only for the call.
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::cuModuleLoadData(&mut raw, image.as_ptr().cast()) }, || {
            format!("cuModuleLoadData({label})")
        })?;
        let mut n = 0;
        ck(unsafe { sys::cuModuleGetFunctionCount(&mut n, raw) }, "cuModuleGetFunctionCount")?;
        let mut fs = vec![std::ptr::null_mut(); n as usize];
        ck(unsafe { sys::cuModuleEnumerateFunctions(fs.as_mut_ptr(), n, raw) }, "cuModuleEnumerateFunctions")?;
        let mut functions = Vec::with_capacity(fs.len());
        for f in fs {
            let mut name: *const c_char = std::ptr::null();
            ck(unsafe { sys::cuFuncGetName(&mut name, f) }, "cuFuncGetName")?;
            functions.push((unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned(), f));
        }
        Ok(Module { raw, label, functions })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.functions.iter().map(|(n, _)| n.as_str())
    }

    /// The kernel whose name is exactly `name`.
    pub fn function(&self, name: &str) -> Option<Kernel> {
        self.functions.iter().find(|(n, _)| n == name).map(|&(_, f)| Kernel { raw: f })
    }

    /// The single kernel whose (possibly mangled) name contains `needle`. Errors list the candidates.
    pub fn find(&self, needle: &str) -> std::result::Result<Kernel, String> {
        let hits: Vec<_> = self.functions.iter().filter(|(n, _)| n.contains(needle)).collect();
        match hits.as_slice() {
            [(_, f)] => Ok(Kernel { raw: *f }),
            [] => Err(format!("{}: no kernel name contains {needle:?}", self.label)),
            many => Err(format!(
                "{}: {} kernels contain {needle:?}: {}",
                self.label,
                many.len(),
                many.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
            )),
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.raw) };
    }
}

/// A kernel handle. Valid while its `Module` lives.
#[derive(Clone, Copy)]
pub struct Kernel {
    raw: sys::CUfunction,
}

#[derive(Clone, Debug)]
pub struct KernelInfo {
    pub name: String,
    pub registers: i32,
    pub local_bytes: i32,
    pub static_shared_bytes: i32,
}

impl Kernel {
    pub fn raw(&self) -> sys::CUfunction {
        self.raw
    }

    pub fn info(&self) -> Result<KernelInfo> {
        let attr = |a| -> Result<i32> {
            let mut v = 0;
            ck(unsafe { sys::cuFuncGetAttribute(&mut v, a, self.raw) }, "cuFuncGetAttribute")?;
            Ok(v)
        };
        let mut name: *const c_char = std::ptr::null();
        ck(unsafe { sys::cuFuncGetName(&mut name, self.raw) }, "cuFuncGetName")?;
        Ok(KernelInfo {
            name: unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned(),
            registers: attr(sys::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            local_bytes: attr(sys::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            static_shared_bytes: attr(sys::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
        })
    }

    /// Allows dynamic shared memory above the default 48 KiB.
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<()> {
        ck(
            unsafe {
                sys::cuFuncSetAttribute(self.raw, sys::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32)
            },
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
        )
    }

    /// Launches with `cuLaunchKernelEx`.
    ///
    /// # Safety
    /// `args` must match the kernel's parameters in count, order, size and meaning, and every device
    /// address in them must stay valid until the kernel completes.
    pub unsafe fn launch(&self, config: &LaunchConfig, stream: &Stream, args: &mut crate::launch::Args) -> Result<()> {
        let mut attr = sys::CUlaunchAttribute { id: 0, pad: [0; 4], value: [0; 64] };
        let mut raw = sys::CUlaunchConfig {
            grid_dim_x: config.grid.0,
            grid_dim_y: config.grid.1,
            grid_dim_z: config.grid.2,
            block_dim_x: config.block.0,
            block_dim_y: config.block.1,
            block_dim_z: config.block.2,
            shared_mem_bytes: config.dynamic_shared_bytes,
            h_stream: stream.raw(),
            attrs: std::ptr::null_mut(),
            num_attrs: 0,
        };
        if config.programmatic_dependent_launch {
            attr.id = sys::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attr.value[..4].copy_from_slice(&1i32.to_ne_bytes());
            raw.attrs = &mut attr;
            raw.num_attrs = 1;
        }
        let mut ptrs = args.pointers();
        ck(
            unsafe {
                sys::cuLaunchKernelEx(&raw, self.raw, ptrs.as_mut_ptr().cast::<*mut c_void>(), std::ptr::null_mut())
            },
            "cuLaunchKernelEx",
        )
    }
}
