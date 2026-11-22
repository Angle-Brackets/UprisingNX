//! the GPU on a thread of its own, so the title's next frame runs while the
//! last one is drawn, as on the console, where the GPU works through its
//! commands while the CPU goes on.
//!
//! the GX commands the title queues (command lists, fills, transfers, DMA),
//! and the register and framebuffer writes that go between them, are jobs the
//! thread runs in order, and the interrupts they raise are handed to the CPU's
//! side when each is done, not when it is queued: a title that waits for the
//! command list's interrupt before reusing its buffers gets it after the GPU
//! stopped reading them.
//!
//! anything else that reaches for the GPU from the CPU's side waits for the
//! thread to finish what it has first, `GpuCell` derefs to the Gpu only then,
//! so every other use of it stays as it was, one thing at a time.
//!
//! ZAKURO_GPU_THREAD=1 turns the thread on. without it the jobs run where
//! they are queued, in the same order, and the interrupts come at once.

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use zakuro_gpu::{Gpu, GpuMemory, Renderer};

use crate::memory::Memory;
use crate::services::gsp::InterruptId;
use crate::GuestMemory;

/// what the GPU is asked to do, in the order it is asked.
pub(crate) enum Gx {
    List { paddr: u32, size: u32 },
    Fill { start: u32, end: u32, value: u32, width: u32 },
    Display { input: u32, output: u32, input_dimensions: u32, output_dimensions: u32, flags: u32 },
    Copy { input: u32, output: u32, size: u32, input_gap: u32, output_gap: u32 },
    Dma { source: u32, dest: u32, size: u32 },
    Framebuffer { screen: u32, active: u32, left: u32, right: u32, stride: u32, format: u32 },
    Register { offset: u32, value: u32 },
    EndFrame,
}

impl Gx {
    /// the kind's bit in ZAKURO_GPU_ASYNC.
    fn bit(&self) -> u32 {
        1 << match self {
            Gx::List { .. } => 0,
            Gx::Display { .. } => 1,
            Gx::Framebuffer { .. } => 2,
            Gx::Register { .. } => 3,
            Gx::EndFrame => 4,
            Gx::Fill { .. } => 5,
            Gx::Copy { .. } => 6,
            Gx::Dma { .. } => 7,
        }
    }
}

/// a command with what it works on and the interrupt that says it is done.
pub(crate) struct Job {
    pub gx: Gx,
    pub interrupt: Option<InterruptId>,
    pub memory: *mut Memory,
    pub renderer: *mut (dyn Renderer + 'static),
    pub linear_base: u32,
}

// SAFETY: the pointers are the system's, which stays where it is while the
// thread lives, and which the CPU's side leaves alone where the thread works,
// see the module's docs. the jobs go to one thread.
unsafe impl Send for Job {}

/// runs a job on the GPU, the interrupt it raises being the job's.
pub(crate) fn execute(gpu: &mut Gpu, job: Job) -> Option<InterruptId> {
    // SAFETY: see Job
    let (memory, renderer) = unsafe { (&mut *job.memory, &mut *job.renderer) };
    let mut guest = GuestMemory { linear_base: job.linear_base, memory };
    // ZAKURO_FRAME_SKIP=<ms>: when what the GPU did for a frame took longer
    // than that, the next frame is thrown away, drawn nowhere and shown
    // never, so the title's logic gets the time back and runs at full speed
    // with the picture updating every other frame. a test, off by default
    static LIMIT: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| std::env::var("ZAKURO_FRAME_SKIP").ok().and_then(|v| v.parse::<f64>().ok()));
    let started = limit.map(|_| std::time::Instant::now());
    let is_end = matches!(job.gx, Gx::EndFrame);
    execute_job(gpu, job.gx, &mut guest, renderer);
    if let (Some(limit), Some(started)) = (limit, started) {
        static BUSY_US: AtomicUsize = AtomicUsize::new(0);
        let spent = started.elapsed().as_micros() as usize;
        let total = (BUSY_US.fetch_add(spent, Ordering::Relaxed) + spent) as f64 / 1000.0;
        if is_end {
            gpu.skip_frame = !gpu.skip_frame && total > limit;
            static ENDS: AtomicUsize = AtomicUsize::new(0);
            static SKIPS: AtomicUsize = AtomicUsize::new(0);
            static MAX_US: AtomicUsize = AtomicUsize::new(0);
            MAX_US.fetch_max((total * 1000.0) as usize, Ordering::Relaxed);
            if gpu.skip_frame {
                SKIPS.fetch_add(1, Ordering::Relaxed);
            }
            if ENDS.fetch_add(1, Ordering::Relaxed) % 120 == 119 {
                log::info!(target: "zakuro_gpu::diag", "frame skip: {} of the last 120 frame ends set it, the longest frame took {:.1} ms", SKIPS.swap(0, Ordering::Relaxed), MAX_US.swap(0, Ordering::Relaxed) as f64 / 1000.0);
            }
            BUSY_US.store(0, Ordering::Relaxed);
        }
    }
    job.interrupt
}

fn execute_job(gpu: &mut Gpu, gx: Gx, guest: &mut GuestMemory<'_>, renderer: &mut dyn Renderer) {
    match gx {
        Gx::List { paddr, size } => gpu.process_command_list(guest, renderer, paddr, size),
        Gx::Fill { start, end, value, width } => gpu.memory_fill(guest, start, end, value, width),
        Gx::Display { input, output, input_dimensions, output_dimensions, flags } => {
            gpu.display_transfer(guest, input, output, input_dimensions, output_dimensions, flags)
        }
        Gx::Copy { input, output, size, input_gap, output_gap } => {
            gpu.texture_copy(guest, input, output, size, input_gap, output_gap)
        }
        Gx::Dma { source, dest, size } => {
            if size >= 0x10000 {
                static DMAS: AtomicUsize = AtomicUsize::new(0);
                if DMAS.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
                    log::info!(target: "zakuro_gpu::diag", "big DMA {source:08x} -> {dest:08x}, {size:x} bytes");
                }
            }
            static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if size < 0x10000 || !*FAST.get_or_init(|| std::env::var_os("ZAKURO_FAST_COPY").is_some()) {
                gpu.sync_memory(guest, source, size);
                gpu.sync_memory(guest, dest, size);
            }
            let mut data = vec![0u8; size as usize];
            guest.read(source, &mut data);
            guest.write(dest, &data);
        }
        Gx::Framebuffer { screen, active, left, right, stride, format } => {
            gpu.set_framebuffer(screen, active, left, right, stride, format)
        }
        Gx::Register { offset, value } => gpu.write_external(offset, value),
        Gx::EndFrame => renderer.end_frame(),
    }
}

struct Shared {
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    /// jobs queued or running.
    pending: AtomicUsize,
    idle: Mutex<()>,
    idle_changed: Condvar,
    /// interrupts of the jobs done, for the CPU's side to raise.
    done: Mutex<Vec<InterruptId>>,
    done_any: AtomicBool,
    quit: AtomicBool,
    gpu: *mut Gpu,
}

// SAFETY: the Gpu pointer is used by the thread alone while jobs are pending,
// and by whoever waited for it to be idle after that.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// the Gpu, and the thread that runs it when there is one.
pub struct GpuCell {
    gpu: Box<UnsafeCell<Gpu>>,
    shared: Arc<Shared>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// what the frontend and the Gpu tell each other, see zakuro_gpu::Flags.
    flags: Arc<zakuro_gpu::Flags>,
    /// internal resolution scale kept here so readers can ask without waiting.
    scale: AtomicU32,
    /// whether frame interpolation is on, kept here to ask without waiting.
    interpolating: AtomicBool,
    /// which kinds of job go to the thread, by Gx::bit, the others run at once
    /// after the thread is done with what it has. ZAKURO_GPU_ASYNC, in hex,
    /// to find which one a title cannot take late; all by default.
    asynchronous: u32,
}

impl GpuCell {
    pub fn new(gpu: Gpu, threaded: bool) -> GpuCell {
        let scale = AtomicU32::new(gpu.scale());
        let flags = gpu.flags.clone();
        let gpu = Box::new(UnsafeCell::new(gpu));
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            pending: AtomicUsize::new(0),
            idle: Mutex::new(()),
            idle_changed: Condvar::new(),
            done: Mutex::new(Vec::new()),
            done_any: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            gpu: gpu.get(),
        });
        let worker = threaded.then(|| {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("gpu".into())
                .stack_size(1 << 20)
                .spawn(move || {
                    zakuro_gpu::thread_role("gpu", 0);
                    work(&shared)
                })
                .expect("the GPU's thread")
        });
        if threaded {
            log::info!("the GPU runs on a thread of its own");
        }
        let asynchronous = std::env::var("ZAKURO_GPU_ASYNC").ok().and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(!0);
        GpuCell { gpu, shared, worker, flags, scale, interpolating: AtomicBool::new(false), asynchronous }
    }

    /// how many times the console's resolution the GPU draws at, without waiting.
    pub fn scale(&self) -> u32 {
        self.scale.load(Ordering::Relaxed)
    }

    /// enables the hardware renderer and updates the cached scale.
    pub fn enable_hardware_renderer(
        &mut self,
        scale: u32,
        device: Option<std::sync::Arc<zakuro_gpu::SharedDevice>>,
        title: Option<u64>,
    ) -> Result<String, String> {
        let res = (&mut **self).enable_hardware_renderer(scale, device, title);
        self.scale.store((&**self).scale(), Ordering::Relaxed);
        res
    }

    /// updates the internal resolution scale and refreshes the cached scale.
    pub fn set_scale<M: zakuro_gpu::GpuMemory>(
        &mut self,
        memory: &mut M,
        scale: u32,
        keep: &[(u32, u32)],
    ) -> u32 {
        let now = (&mut **self).set_scale(memory, scale, keep);
        self.scale.store(now, Ordering::Relaxed);
        now
    }

    /// whether the GPU has a thread.
    pub fn threaded(&self) -> bool {
        self.worker.is_some()
    }

    /// what the frontend and the Gpu tell each other, to use without waiting.
    pub fn flags(&self) -> &Arc<zakuro_gpu::Flags> {
        &self.flags
    }

    /// whether frame interpolation is on, without waiting.
    pub fn interpolating(&self) -> bool {
        self.interpolating.load(Ordering::Relaxed)
    }

    pub(crate) fn set_interpolating(&self, on: bool) {
        self.interpolating.store(on, Ordering::Relaxed);
    }

    /// queues a job, or runs it, and gives the interrupt if it ran. else it
    /// comes with take_done.
    pub(crate) fn submit(&self, job: Job) -> Option<InterruptId> {
        // what titles take to be done at once, DMA, fills and copies, is, after
        // what the thread has to do before it: they wait for it and go on
        // from here, as the thread would run them, and their interrupt is
        // the CPU's side to give at once
        let at_once = matches!(job.gx, Gx::Dma { .. } | Gx::Fill { .. } | Gx::Copy { .. })
            || self.asynchronous & job.gx.bit() == 0;
        if self.worker.is_none() || at_once {
            self.wait_idle();
            // SAFETY: no thread at work, and the caller holds the only handle
            return execute(unsafe { &mut *self.gpu.get() }, job);
        }
        self.shared.pending.fetch_add(1, Ordering::AcqRel);
        self.shared.queue.lock().unwrap_or_else(|p| p.into_inner()).push_back(job);
        self.shared.wake.notify_one();
        None
    }

    /// whether jobs are queued or running.
    pub fn busy(&self) -> bool {
        self.shared.pending.load(Ordering::Acquire) != 0
    }

    /// waits for the thread to finish every job queued.
    pub fn wait_idle(&self) {
        self.wait_idle_as(2);
    }

    /// the same, counted under who waits, see zakuro_gpu::stats::BARRIER_NAMES.
    pub fn wait_idle_as(&self, who: usize) {
        if self.shared.pending.load(Ordering::Acquire) == 0 {
            return;
        }
        let started = std::time::Instant::now();
        self.wait_idle_slow();
        zakuro_gpu::stats::BARRIER_NS[who].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        zakuro_gpu::stats::BARRIER_COUNT[who].fetch_add(1, Ordering::Relaxed);
    }

    fn wait_idle_slow(&self) {
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            // the CPU's side has nothing to do but wait, and shades vertices
            // for the thread meanwhile, see zakuro_gpu::shade_queue
            if zakuro_gpu::shade_queue::help() {
                continue;
            }
            let guard = self.shared.idle.lock().unwrap_or_else(|p| p.into_inner());
            if self.shared.pending.load(Ordering::Acquire) == 0 {
                break;
            }
            let _ = self.shared.idle_changed.wait_timeout(guard, std::time::Duration::from_micros(200));
        }
    }

    /// waits for a job to finish that has an interrupt to give, or for none
    /// to be left.
    pub fn wait_done(&self) {
        let started = std::time::Instant::now();
        self.wait_done_slow();
        zakuro_gpu::stats::BARRIER_NS[0].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        zakuro_gpu::stats::BARRIER_COUNT[0].fetch_add(1, Ordering::Relaxed);
    }

    fn wait_done_slow(&self) {
        while !self.shared.done_any.load(Ordering::Acquire) && self.shared.pending.load(Ordering::Acquire) != 0 {
            if zakuro_gpu::shade_queue::help() {
                continue;
            }
            let guard = self.shared.idle.lock().unwrap_or_else(|p| p.into_inner());
            if self.shared.done_any.load(Ordering::Acquire) || self.shared.pending.load(Ordering::Acquire) == 0 {
                break;
            }
            let _ = self.shared.idle_changed.wait_timeout(guard, std::time::Duration::from_micros(200));
        }
    }

    /// jobs queued or running, and interrupts done and not raised yet, for logs.
    pub fn state(&self) -> (usize, usize) {
        (
            self.shared.pending.load(Ordering::Acquire),
            self.shared.done.lock().unwrap_or_else(|p| p.into_inner()).len(),
        )
    }

    /// whether finished GPU interrupts are waiting to be raised.
    pub fn has_done(&self) -> bool {
        self.shared.done_any.load(Ordering::Acquire)
    }

    /// the interrupts of the jobs done since this was last asked, in order.
    pub fn take_done(&self) -> Vec<InterruptId> {
        if !self.shared.done_any.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut done = self.shared.done.lock().unwrap_or_else(|p| p.into_inner());
        self.shared.done_any.store(false, Ordering::Release);
        std::mem::take(&mut *done)
    }
}

impl Deref for GpuCell {
    type Target = Gpu;

    fn deref(&self) -> &Gpu {
        self.wait_idle();
        // SAFETY: the thread is idle, and takes no job until the CPU's side
        // queues one, which it cannot while this is borrowed
        unsafe { &*self.gpu.get() }
    }
}

impl DerefMut for GpuCell {
    fn deref_mut(&mut self) -> &mut Gpu {
        self.wait_idle();
        // SAFETY: as in deref
        unsafe { &mut *self.gpu.get() }
    }
}

impl Drop for GpuCell {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.wait_idle();
            // Hold `queue` while setting `quit` and notifying so `work()` cannot
            // observe `quit == false` and then enter `wake.wait()` after our notify.
            let _queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            self.shared.quit.store(true, Ordering::Release);
            self.shared.wake.notify_all();
            drop(_queue);
            let _ = worker.join();
        }
    }
}

#[cfg(target_os = "horizon")]
extern "C" {
    /// the Switch frontend's profiler, which samples the threads that ask.
    fn prof_add_thread();
}

fn work(shared: &Shared) {
    #[cfg(target_os = "horizon")]
    // SAFETY: the frontend's, which has it linked in
    unsafe {
        prof_add_thread()
    };
    loop {
        let job = {
            let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                if shared.quit.load(Ordering::Acquire) {
                    return;
                }
                queue = shared.wake.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        // SAFETY: while jobs are pending the CPU's side does not touch it
        let interrupt = execute(unsafe { &mut *shared.gpu }, job);
        if let Some(interrupt) = interrupt {
            let mut done = shared.done.lock().unwrap_or_else(|p| p.into_inner());
            done.push(interrupt);
            shared.done_any.store(true, Ordering::Release);
        }
        {
            let _idle = shared.idle.lock().unwrap_or_else(|p| p.into_inner());
            shared.pending.fetch_sub(1, Ordering::AcqRel);
        }
        shared.idle_changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_does_not_lose_wakeup() {
        for _ in 0..64 {
            let cell = GpuCell::new(Gpu::new(), true);
            drop(cell);
        }
    }
}
