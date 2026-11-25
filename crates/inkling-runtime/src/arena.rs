//! One device allocation carved into named, aligned slots.
//!
//! Plan every slot first (`ArenaPlan::reserve`), then allocate once (`ArenaPlan::allocate`). The arena's
//! size is exactly the planned span, so nothing is allocated that no slot uses, and every slot keeps the
//! name it was planned under for error messages and memory reports.

use crate::error::Result;
use crate::memory::{DevSlice, DeviceBuffer};

/// Default slot alignment: 256 bytes covers 128-bit vector loads and `cp.async` / TMA requirements.
pub const DEFAULT_ALIGN: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot(usize);

#[derive(Clone, Debug)]
pub struct SlotInfo {
    pub name: String,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Default)]
pub struct ArenaPlan {
    slots: Vec<SlotInfo>,
    end: usize,
}

fn align_up(x: usize, align: usize) -> usize {
    x.div_ceil(align) * align
}

impl ArenaPlan {
    pub fn new() -> ArenaPlan {
        ArenaPlan::default()
    }

    /// Reserves `bytes` at the next multiple of `align` (a power of two).
    pub fn reserve_aligned(&mut self, name: impl Into<String>, bytes: usize, align: usize) -> Slot {
        assert!(align.is_power_of_two(), "alignment {align} is not a power of two");
        let offset = align_up(self.end, align);
        self.end = offset.checked_add(bytes).expect("arena size overflows usize");
        self.slots.push(SlotInfo { name: name.into(), offset, bytes });
        Slot(self.slots.len() - 1)
    }

    pub fn reserve(&mut self, name: impl Into<String>, bytes: usize) -> Slot {
        self.reserve_aligned(name, bytes, DEFAULT_ALIGN)
    }

    /// Reserves `count` elements of `T`.
    pub fn reserve_of<T>(&mut self, name: impl Into<String>, count: usize) -> Slot {
        self.reserve(name, count.checked_mul(size_of::<T>()).expect("slot size overflows usize"))
    }

    /// Total bytes the arena will occupy.
    pub fn total_bytes(&self) -> usize {
        align_up(self.end, DEFAULT_ALIGN)
    }

    pub fn slots(&self) -> &[SlotInfo] {
        &self.slots
    }

    /// Allocates the planned span in one `cuMemAlloc`.
    pub fn allocate(self) -> Result<Arena> {
        let total = self.total_bytes();
        let buffer = DeviceBuffer::alloc(total)?;
        Ok(Arena { buffer, slots: self.slots, total })
    }
}

pub struct Arena {
    buffer: DeviceBuffer,
    slots: Vec<SlotInfo>,
    total: usize,
}

impl Arena {
    pub fn get(&self, slot: Slot) -> DevSlice {
        let s = &self.slots[slot.0];
        self.buffer.slice().sub(s.offset, s.bytes)
    }

    pub fn info(&self, slot: Slot) -> &SlotInfo {
        &self.slots[slot.0]
    }

    pub fn whole(&self) -> DevSlice {
        self.buffer.slice()
    }

    pub fn total_bytes(&self) -> usize {
        self.total
    }

    pub fn slots(&self) -> &[SlotInfo] {
        &self.slots
    }

    /// Bytes that belong to no slot (alignment padding only). The arena is "exactly filled" when this is
    /// less than one alignment unit per slot.
    pub fn padding_bytes(&self) -> usize {
        self.total - self.slots.iter().map(|s| s.bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_aligned_disjoint_and_cover_the_span() {
        let mut p = ArenaPlan::new();
        let a = p.reserve("a", 10);
        let b = p.reserve_of::<u32>("b", 100);
        let c = p.reserve_aligned("c", 1, 4096);
        let s = p.slots();
        assert_eq!((s[a.0].offset, s[b.0].offset, s[c.0].offset), (0, 256, 4096));
        assert!(s.windows(2).all(|w| w[0].offset + w[0].bytes <= w[1].offset));
        assert_eq!(p.total_bytes(), 4096 + 256);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two_alignment() {
        ArenaPlan::new().reserve_aligned("x", 1, 3);
    }
}
