//! Stream capture into CUDA graphs and replay.

use crate::error::{Result, ck};
use crate::stream::Stream;
use crate::sys;

/// Captures everything `record` enqueues on `stream` (thread-local mode) into an instantiated graph.
/// On a capture error the stream's capture is ended before returning, so the stream stays usable.
pub fn capture<E>(stream: &Stream, record: impl FnOnce(&Stream) -> std::result::Result<(), E>) -> std::result::Result<GraphExec, E>
where
    E: From<crate::error::CudaError>,
{
    ck(
        unsafe { sys::cuStreamBeginCapture_v2(stream.raw(), sys::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
        "cuStreamBeginCapture",
    )?;
    let recorded = record(stream);
    let mut graph = std::ptr::null_mut();
    let ended = ck(unsafe { sys::cuStreamEndCapture(stream.raw(), &mut graph) }, "cuStreamEndCapture");
    recorded?;
    ended?;
    let graph = Graph { raw: graph };
    Ok(graph.instantiate()?)
}

/// Builds an executable graph whose middle is a switch conditional node (CUDA 12.4+ device-side
/// branching), as squid5090's adaptive verify graph:
/// 1. `head(stream, handle)` is captured into the root graph; it must launch a kernel that calls
///    `cudaGraphSetConditional(handle, case)`.
/// 2. A switch node with `cases` bodies follows it; `tail(stream)` is captured after the switch (host
///    copies of the result, for example).
/// 3. `body(i, stream)` is captured into case `i`. Bodies may contain kernels, memcpys and memsets
///    (multi-stream fork/join inside a body becomes plain edges), not event or host nodes.
pub fn capture_switch<E>(
    stream: &Stream,
    cases: u32,
    head: impl FnOnce(&Stream, sys::CUgraphConditionalHandle) -> std::result::Result<(), E>,
    tail: impl FnOnce(&Stream) -> std::result::Result<(), E>,
    mut body: impl FnMut(u32, &Stream) -> std::result::Result<(), E>,
) -> std::result::Result<GraphExec, E>
where
    E: From<crate::error::CudaError>,
{
    let mut root = std::ptr::null_mut();
    ck(unsafe { sys::cuGraphCreate(&mut root, 0) }, "cuGraphCreate")?;
    let root = Graph { raw: root };
    let mut ctx = std::ptr::null_mut();
    ck(unsafe { sys::cuCtxGetCurrent(&mut ctx) }, "cuCtxGetCurrent")?;
    let mut handle = 0;
    ck(unsafe { sys::cuGraphConditionalHandleCreate(&mut handle, root.raw, ctx, 0, 0) }, "cuGraphConditionalHandleCreate")?;

    let begin = |g: sys::CUgraph| {
        ck(
            unsafe {
                sys::cuStreamBeginCaptureToGraph(stream.raw(), g, std::ptr::null(), std::ptr::null(), 0, sys::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            },
            "cuStreamBeginCaptureToGraph",
        )
    };
    let end = || {
        let mut g = std::ptr::null_mut();
        ck(unsafe { sys::cuStreamEndCapture(stream.raw(), &mut g) }, "cuStreamEndCapture")
    };

    begin(root.raw)?;
    let mut bodies: Vec<sys::CUgraph> = Vec::new();
    let head_result = (|| -> std::result::Result<(), E> {
        head(stream, handle)?;
        let (mut status, mut id, mut g, mut deps, mut edges, mut n) =
            (0i32, 0u64, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 0usize);
        ck(
            unsafe { sys::cuStreamGetCaptureInfo_v3(stream.raw(), &mut status, &mut id, &mut g, &mut deps, &mut edges, &mut n) },
            "cuStreamGetCaptureInfo",
        )?;
        let mut p = sys::CUgraphNodeParamsConditional {
            node_type: sys::CU_GRAPH_NODE_TYPE_CONDITIONAL,
            reserved0: [0; 3],
            handle,
            cond_type: sys::CU_GRAPH_COND_TYPE_SWITCH,
            size: cases,
            ph_graph_out: std::ptr::null_mut(),
            ctx,
            pad: [0; 200],
            reserved2: 0,
        };
        let mut node = std::ptr::null_mut();
        ck(unsafe { sys::cuGraphAddNode_v2(&mut node, g, deps, edges, n, &mut p) }, "cuGraphAddNode(switch)")?;
        bodies = (0..cases as usize).map(|i| unsafe { *p.ph_graph_out.add(i) }).collect();
        ck(
            unsafe { sys::cuStreamUpdateCaptureDependencies_v2(stream.raw(), &mut node, std::ptr::null(), 1, sys::CU_STREAM_SET_CAPTURE_DEPENDENCIES) },
            "cuStreamUpdateCaptureDependencies",
        )?;
        tail(stream)
    })();
    let ended = end();
    head_result?;
    ended?;
    for (i, &b) in bodies.iter().enumerate() {
        begin(b)?;
        let r = body(i as u32, stream);
        let ended = end();
        r?;
        ended?;
    }
    Ok(root.instantiate()?)
}

struct Graph {
    raw: sys::CUgraph,
}

impl Graph {
    fn node_count(&self) -> Result<usize> {
        let mut n = 0;
        ck(unsafe { sys::cuGraphGetNodes(self.raw, std::ptr::null_mut(), &mut n) }, "cuGraphGetNodes")?;
        Ok(n)
    }

    fn instantiate(self) -> Result<GraphExec> {
        let nodes = self.node_count()?;
        let mut raw = std::ptr::null_mut();
        ck(unsafe { sys::cuGraphInstantiateWithFlags(&mut raw, self.raw, 0) }, "cuGraphInstantiate")?;
        Ok(GraphExec { raw, nodes })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::cuGraphDestroy(self.raw) };
    }
}

/// An executable graph.
pub struct GraphExec {
    raw: sys::CUgraphExec,
    nodes: usize,
}

unsafe impl Send for GraphExec {}

impl GraphExec {
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Uploads the graph's work descriptors ahead of the first launch.
    pub fn upload(&self, stream: &Stream) -> Result<()> {
        ck(unsafe { sys::cuGraphUpload(self.raw, stream.raw()) }, "cuGraphUpload")
    }

    pub fn launch(&self, stream: &Stream) -> Result<()> {
        ck(unsafe { sys::cuGraphLaunch(self.raw, stream.raw()) }, "cuGraphLaunch")
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { sys::cuGraphExecDestroy(self.raw) };
    }
}
