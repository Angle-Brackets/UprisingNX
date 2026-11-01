//! Probes the std features Zakuro uses, on Horizon. Every check prints
//! `probe <name>: ok ...` or `probe <name>: FAIL ...`, so a crash shows up as
//! the last line printed.

use std::cell::Cell;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

thread_local! {
    static COUNTER: Cell<u64> = const { Cell::new(0) };
}

fn say(line: &str) {
    println!("{line}");
    let _ = std::io::stdout().flush();
}

fn tls_bump(n: u64) -> u64 {
    for _ in 0..n {
        COUNTER.with(|c| c.set(c.get() + 1));
    }
    COUNTER.with(|c| c.get())
}

fn check_threads() {
    // each spawned thread must see its own thread-local, starting at 0
    let handles: Vec<_> = (0..4u64)
        .map(|i| {
            std::thread::Builder::new()
                .name(format!("probe-{i}"))
                .stack_size(256 * 1024)
                .spawn(move || (std::thread::current().name().map(str::to_owned), tls_bump(1000 + i)))
                .expect("spawn")
        })
        .collect();
    let mut ok = true;
    for (i, h) in handles.into_iter().enumerate() {
        let (name, count) = h.join().expect("join");
        ok &= count == 1000 + i as u64 && name.as_deref() == Some(&*format!("probe-{i}"));
    }
    let main = tls_bump(5);
    say(&format!("probe thread+tls: {} (main tls {main})", if ok && main == 5 { "ok" } else { "FAIL" }));
}

fn check_sync() {
    let pair = Arc::new((Mutex::new(0u32), Condvar::new()));
    let (tx, rx) = mpsc::channel::<u32>();
    let p2 = pair.clone();
    let worker = std::thread::spawn(move || {
        for v in rx {
            let (m, cv) = &*p2;
            *m.lock().unwrap() += v;
            cv.notify_one();
        }
    });
    for v in 1..=100 {
        tx.send(v).unwrap();
    }
    let (m, cv) = &*pair;
    let guard = cv
        .wait_timeout_while(m.lock().unwrap(), Duration::from_secs(5), |s| *s != 5050)
        .unwrap()
        .0;
    let sum = *guard;
    drop(guard);
    drop(tx);
    worker.join().unwrap();
    say(&format!("probe mutex+condvar+mpsc: {} (sum {sum})", if sum == 5050 { "ok" } else { "FAIL" }));
}

fn check_rayon() {
    use rayon::prelude::*;
    let par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    say(&format!("probe available_parallelism: {par}"));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build();
    let pool = match pool {
        Ok(p) => p,
        Err(e) => return say(&format!("probe rayon: FAIL build pool: {e}")),
    };
    let v: Vec<u64> = (0..2_000_000).collect();
    let t = Instant::now();
    let s: u64 = pool.install(|| v.par_iter().map(|x| x * 3 + 1).sum());
    let par_t = t.elapsed();
    let t = Instant::now();
    let s2: u64 = v.iter().map(|x| x * 3 + 1).sum();
    let ser_t = t.elapsed();
    let threads = AtomicU64::new(0);
    pool.broadcast(|_| threads.fetch_add(1, Ordering::Relaxed));
    say(&format!(
        "probe rayon: {} (par {:?}, serial {:?}, workers {})",
        if s == s2 { "ok" } else { "FAIL" },
        par_t,
        ser_t,
        threads.load(Ordering::Relaxed)
    ));
}

fn check_time() {
    let t = Instant::now();
    std::thread::sleep(Duration::from_millis(50));
    let e = t.elapsed();
    let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let ok = e >= Duration::from_millis(45) && e < Duration::from_millis(200);
    say(&format!("probe instant+sleep: {} (50 ms sleep took {e:?}, unix time {wall})", if ok { "ok" } else { "FAIL" }));
}

fn check_fs() {
    let dir = "sdmc:/switch/uprisingnx/probe";
    let r = (|| -> std::io::Result<String> {
        std::fs::create_dir_all(dir)?;
        let path = format!("{dir}/hello.bin");
        let data: Vec<u8> = (0..1_000_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &data)?;
        let back = std::fs::read(&path)?;
        if back != data {
            return Ok("FAIL readback differs".into());
        }
        // Zakuro opens the ROM once and reads at offsets: check seek+read
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::Start(123_457))?;
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        let seek_ok = b == [(123_457u32) as u8, 123_458u32 as u8, 123_459u32 as u8, 123_460u32 as u8];
        let names: Vec<String> = match std::fs::read_dir(dir) {
            Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect(),
            Err(e) => vec![format!("read_dir error: {e}")],
        };
        Ok(format!("{} (seek {}, read_dir {names:?})", if seek_ok { "ok" } else { "FAIL" }, seek_ok))
    })();
    match r {
        Ok(s) => say(&format!("probe fs: {s}")),
        Err(e) => say(&format!("probe fs: FAIL {e}")),
    }
}

fn check_alloc() {
    // Zakuro boxes FCRAM/VRAM up front; the recompiled code is ~435 MiB on top
    for mb in [128usize, 512, 1024] {
        let t = Instant::now();
        let v = vec![0u8; mb << 20];
        let ok = v.len() == mb << 20;
        say(&format!("probe alloc {mb} MiB: {} ({:?})", if ok { "ok" } else { "FAIL" }, t.elapsed()));
        drop(v);
    }
}

#[no_mangle]
pub extern "C" fn std_probe_run() {
    say("probe start");
    check_time();
    check_threads();
    check_sync();
    check_rayon();
    check_fs();
    check_alloc();
    say("probe done");
}
