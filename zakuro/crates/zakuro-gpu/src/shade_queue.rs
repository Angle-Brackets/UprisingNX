//! a queue of vertex shading jobs, which helper threads and, while it has
//! nothing else to do, the game's thread work through, so the GPU's thread
//! can go on to the next draw while the vertices of the ones before are
//! shaded, see raster::draw.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send>;

struct Queue {
    jobs: Mutex<VecDeque<Job>>,
    wake: Condvar,
}

fn queue() -> &'static Queue {
    static QUEUE: OnceLock<&'static Queue> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let queue: &'static Queue = Box::leak(Box::new(Queue { jobs: Mutex::new(VecDeque::new()), wake: Condvar::new() }));
        let helpers = std::env::var("ZAKURO_SHADE_HELPERS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
        for index in 0..helpers {
            std::thread::Builder::new()
                .name(format!("shade helper {index}"))
                .stack_size(256 * 1024)
                .spawn(move || {
                    crate::thread_role("shade", index);
                    loop {
                    let job = {
                        let mut jobs = queue.jobs.lock().unwrap_or_else(|p| p.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = queue.wake.wait(jobs).unwrap_or_else(|p| p.into_inner());
                        }
                    };
                    job();
                    }
                })
                .expect("a shading helper");
        }
        queue
    })
}

/// has a helper shade, or the game's thread when it helps.
pub(crate) fn submit(job: Job) {
    let queue = queue();
    queue.jobs.lock().unwrap_or_else(|p| p.into_inner()).push_back(job);
    queue.wake.notify_one();
}

/// runs one waiting job on this thread, if there is one, for a thread with
/// nothing to do. whether it did.
pub fn help() -> bool {
    // ZAKURO_GAME_HELPS=0 keeps the game's thread out of it
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("ZAKURO_GAME_HELPS").map_or(true, |v| v != "0")) {
        return false;
    }
    let queue = queue();
    let job = queue.jobs.lock().unwrap_or_else(|p| p.into_inner()).pop_front();
    match job {
        Some(job) => {
            job();
            true
        }
        None => false,
    }
}

thread_local! {
    /// set on a thread running a job, whose draw's vertices the pool must
    /// not be asked to split again.
    pub(crate) static IN_JOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
