//! Launch geometry and kernel argument packing.

use crate::memory::DevSlice;
use std::ffi::c_void;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    /// Launch with programmatic stream serialization (PDL): the kernel may start before its predecessor
    /// finishes and must call the grid-dependency wait before reading the predecessor's outputs.
    pub programmatic_dependent_launch: bool,
}

impl LaunchConfig {
    pub fn new(grid: impl Into<Dim>, block: impl Into<Dim>) -> LaunchConfig {
        LaunchConfig {
            grid: grid.into().0,
            block: block.into().0,
            dynamic_shared_bytes: 0,
            programmatic_dependent_launch: false,
        }
    }

    pub fn shared(mut self, bytes: u32) -> LaunchConfig {
        self.dynamic_shared_bytes = bytes;
        self
    }

    pub fn pdl(mut self, enabled: bool) -> LaunchConfig {
        self.programmatic_dependent_launch = enabled;
        self
    }
}

pub struct Dim(pub (u32, u32, u32));

impl From<u32> for Dim {
    fn from(x: u32) -> Dim {
        Dim((x, 1, 1))
    }
}

impl From<(u32, u32)> for Dim {
    fn from((x, y): (u32, u32)) -> Dim {
        Dim((x, y, 1))
    }
}

impl From<(u32, u32, u32)> for Dim {
    fn from(d: (u32, u32, u32)) -> Dim {
        Dim(d)
    }
}

/// A scalar kernel parameter: stored by value, passed to the driver by address.
pub trait KernelArg: Copy {
    const SIZE: usize;
    fn write(&self, out: &mut [u8]);
}

macro_rules! scalar_arg {
    ($($t:ty),*) => {$(
        impl KernelArg for $t {
            const SIZE: usize = size_of::<$t>();
            fn write(&self, out: &mut [u8]) {
                out[..Self::SIZE].copy_from_slice(&self.to_ne_bytes());
            }
        }
    )*};
}
scalar_arg!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64, usize);

/// C++ `bool` kernel parameters are one byte.
impl KernelArg for bool {
    const SIZE: usize = 1;
    fn write(&self, out: &mut [u8]) {
        out[0] = *self as u8;
    }
}

impl KernelArg for DevSlice {
    const SIZE: usize = 8;
    fn write(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.ptr.to_ne_bytes());
    }
}

/// A by-value aggregate of device pointers, e.g. squid5090's `Ptr3`/`MutPtr3`/`HalfPtr3`/`BF16Ptr3`
/// (`[u64; 3]`, 24 bytes) or `CtrlPtr3` (`[u64; 9]`, 72 bytes). Null pointers are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtrPack<const N: usize>(pub [u64; N]);

impl<const N: usize> KernelArg for PtrPack<N> {
    const SIZE: usize = 8 * N;
    fn write(&self, out: &mut [u8]) {
        for (i, p) in self.0.iter().enumerate() {
            out[8 * i..8 * i + 8].copy_from_slice(&p.to_ne_bytes());
        }
    }
}

/// Kernel parameters in C order. Each value gets its own 8-byte-aligned cell, so the addresses handed to
/// the driver stay valid and aligned for every type used here (scalars, pointers, pointer packs).
#[derive(Default)]
pub struct Args {
    cells: Vec<Box<[u64]>>,
}

impl Args {
    pub fn new() -> Args {
        Args::default()
    }

    pub fn push<T: KernelArg>(&mut self, v: T) -> &mut Args {
        let mut cell = vec![0u64; T::SIZE.div_ceil(8).max(1)].into_boxed_slice();
        let bytes = unsafe { std::slice::from_raw_parts_mut(cell.as_mut_ptr().cast::<u8>(), cell.len() * 8) };
        v.write(bytes);
        self.cells.push(cell);
        self
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub(crate) fn pointers(&mut self) -> Vec<*mut c_void> {
        self.cells.iter_mut().map(|c| c.as_mut_ptr().cast::<c_void>()).collect()
    }
}

/// `args![a, b, c]` builds `Args` from values implementing `KernelArg`.
#[macro_export]
macro_rules! args {
    ($($v:expr),* $(,)?) => {{
        let mut a = $crate::launch::Args::new();
        $( a.push($v); )*
        a
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_store_values_by_address() {
        let mut a = Args::new();
        a.push(7u32)
            .push(-2.5f32)
            .push(DevSlice { ptr: 0xdead_beef, bytes: 4 })
            .push(true)
            .push(PtrPack([1u64, 0, 3, 4, 5, 6, 7, 8, 9]));
        let ptrs = a.pointers();
        unsafe {
            assert_eq!(*(ptrs[0] as *const u32), 7);
            assert_eq!(*(ptrs[1] as *const f32), -2.5);
            assert_eq!(*(ptrs[2] as *const u64), 0xdead_beef);
            assert_eq!(*(ptrs[3] as *const u8), 1);
            let pack = std::slice::from_raw_parts(ptrs[4] as *const u64, 9);
            assert_eq!(pack, [1, 0, 3, 4, 5, 6, 7, 8, 9]);
        }
    }
}
