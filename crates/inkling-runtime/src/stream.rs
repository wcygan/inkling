use crate::error::{Result, ck};
use crate::sys;

/// A non-blocking stream.
pub struct Stream {
    raw: sys::CUstream,
}

unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut raw = std::ptr::null_mut();
        ck(unsafe { sys::cuStreamCreate(&mut raw, sys::CU_STREAM_NON_BLOCKING) }, "cuStreamCreate")?;
        Ok(Stream { raw })
    }

    pub fn raw(&self) -> sys::CUstream {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        ck(unsafe { sys::cuStreamSynchronize(self.raw) }, "cuStreamSynchronize")
    }

    /// Makes later work on this stream wait for `event`.
    pub fn wait(&self, event: &Event) -> Result<()> {
        ck(unsafe { sys::cuStreamWaitEvent(self.raw, event.raw, 0) }, "cuStreamWaitEvent")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::cuStreamDestroy_v2(self.raw) };
    }
}

pub struct Event {
    raw: sys::CUevent,
}

unsafe impl Send for Event {}

impl Event {
    /// A timing event.
    pub fn new() -> Result<Event> {
        Event::with_flags(sys::CU_EVENT_DEFAULT)
    }

    /// A cheaper event for ordering and completion checks only.
    pub fn new_untimed() -> Result<Event> {
        Event::with_flags(sys::CU_EVENT_DISABLE_TIMING)
    }

    fn with_flags(flags: u32) -> Result<Event> {
        let mut raw = std::ptr::null_mut();
        ck(unsafe { sys::cuEventCreate(&mut raw, flags) }, "cuEventCreate")?;
        Ok(Event { raw })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        ck(unsafe { sys::cuEventRecord(self.raw, stream.raw()) }, "cuEventRecord")
    }

    pub fn synchronize(&self) -> Result<()> {
        ck(unsafe { sys::cuEventSynchronize(self.raw) }, "cuEventSynchronize")
    }

    /// Milliseconds from `start` to `self`; both must be timing events that have completed.
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0.0;
        ck(unsafe { sys::cuEventElapsedTime_v2(&mut ms, start.raw, self.raw) }, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::cuEventDestroy_v2(self.raw) };
    }
}
