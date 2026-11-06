//! a memory search for finding where a game keeps a value (health, say),
//! and pins to hold a value there, for testing. off unless ZAKURO_SCAN names
//! a folder; the commands come as one line in scan.cmd there, the answer
//! goes to scan.out.
//!
//!   snap          every word of guest memory is a candidate
//!   decu incu     keep the words whose value went down or up (integers)
//!   decf incf     the same, the words read as floats
//!   same diff     keep the words that stayed or changed
//!   equu N        keep the words equal to the integer N
//!   equf X        keep the words equal to the float X
//!   pin ADDR HEX  write the word HEX at ADDR every frame
//!   unpin         drop every pin
//!   undo          go back before the last filter (small lists only)

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::System;

struct Scan {
    dir: PathBuf,
    frame: u64,
    found: Vec<(u32, u32)>,
    history: Vec<Vec<(u32, u32)>>,
    pins: Vec<(u32, u32)>,
}

static SCAN: OnceLock<Option<Mutex<Scan>>> = OnceLock::new();

const RANGES: [(u32, u32); 4] = [(0x0010_0000, 0x00F0_0000), (0x0800_0000, 0x0800_0000), (0x1400_0000, 0x0800_0000), (0x3000_0000, 0x1000_0000)];

fn state() -> Option<&'static Mutex<Scan>> {
    SCAN.get_or_init(|| {
        let dir = std::env::var_os("ZAKURO_SCAN")?;
        log::info!("memory search on, commands in {}/scan.cmd", PathBuf::from(&dir).display());
        Some(Mutex::new(Scan { dir: dir.into(), frame: 0, found: Vec::new(), history: Vec::new(), pins: Vec::new() }))
    })
    .as_ref()
}

/// every mapped, readable word as (address, value).
fn all_words(system: &mut System) -> Vec<(u32, u32)> {
    let mut words = Vec::new();
    for (start, size) in RANGES {
        let end = start as u64 + size as u64;
        let mut at = start as u64;
        while at < end {
            let info = system.memory.query(at as u32);
            let next = (info.base as u64 + info.size as u64).max(at + 0x1000);
            if info.state != 0 && info.permission & 1 != 0 {
                let from = at.max(info.base as u64);
                let to = next.min(end);
                let mut bytes = vec![0u8; (to - from) as usize];
                system.memory.read_bytes(from as u32, &mut bytes);
                for (i, word) in bytes.chunks_exact(4).enumerate() {
                    words.push((from as u32 + i as u32 * 4, u32::from_le_bytes([word[0], word[1], word[2], word[3]])));
                }
            }
            at = next;
        }
    }
    words
}

pub fn tick(system: &mut System) {
    let Some(state) = state() else { return };
    let mut scan = state.lock().unwrap();
    scan.frame += 1;
    for index in 0..scan.pins.len() {
        let (address, value) = scan.pins[index];
        system.memory.write_bytes(address, &value.to_le_bytes());
    }
    if scan.frame % 20 != 0 {
        return;
    }
    let cmd = scan.dir.join("scan.cmd");
    let Ok(text) = std::fs::read_to_string(&cmd) else { return };
    let _ = std::fs::remove_file(&cmd);
    let parts: Vec<&str> = text.split_whitespace().collect();
    let mut reply = String::new();
    match parts.as_slice() {
        ["snap"] => scan.found = all_words(system),
        ["pin", address, value] => {
            let address = u32::from_str_radix(address.trim_start_matches("0x"), 16).unwrap_or(0);
            let value = u32::from_str_radix(value.trim_start_matches("0x"), 16).unwrap_or(0);
            scan.pins.push((address, value));
        }
        ["unpin"] => scan.pins.clear(),
        ["dump", address, count] => {
            let address = u32::from_str_radix(address.trim_start_matches("0x"), 16).unwrap_or(0);
            let count = count.parse::<usize>().unwrap_or(16).min(4096);
            let mut bytes = vec![0u8; count * 4];
            system.memory.read_bytes(address, &mut bytes);
            let mut out = String::new();
            for (i, word) in bytes.chunks_exact(4).enumerate() {
                let value = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                out.push_str(&format!("{:08X} {value:08X} {} {}\n", address + i as u32 * 4, value, f32::from_bits(value)));
            }
            let _ = std::fs::write(scan.dir.join("dump.out"), out);
        }
        ["undo"] => {
            if let Some(before) = scan.history.pop() {
                scan.found = before;
            }
        }
        [kind, rest @ ..] => {
            let now = all_words(system);
            let live: std::collections::HashMap<u32, u32> = now.into_iter().collect();
            let argument = rest.first().copied().unwrap_or("");
            let target_u = argument.parse::<u32>().unwrap_or(0);
            let target_f = argument.parse::<f32>().unwrap_or(0.0);
            let kind = *kind;
            let old = std::mem::take(&mut scan.found);
            if old.len() < 3_000_000 {
                scan.history.push(old.clone());
            }
            scan.found = old
                .into_iter()
                .filter_map(|(address, before)| {
                    let after = *live.get(&address)?;
                    let (b, a) = (f32::from_bits(before), f32::from_bits(after));
                    let keep = match kind {
                        "decu" => after < before,
                        "incu" => after > before,
                        "decf" => a < b && a.is_finite() && b.is_finite(),
                        "incf" => a > b && a.is_finite() && b.is_finite(),
                        "same" => after == before,
                        "diff" => after != before,
                        "equu" => after == target_u,
                        "ltu" => after < target_u,
                        "gtu" => after > target_u,
                        "ltf" => a < target_f && a.is_finite(),
                        "gtf" => a > target_f && a.is_finite(),
                        "show" => true,
                        "equf" => (a - target_f).abs() < 1e-4,
                        _ => true,
                    };
                    keep.then_some((address, after))
                })
                .collect();
        }
        [] => {}
    }
    reply.push_str(&format!("{} candidates after {:?}\n", scan.found.len(), text.trim()));
    if scan.found.len() <= 3000 {
        for (address, value) in &scan.found {
            reply.push_str(&format!("{address:08X} {value:08X} {} {}\n", *value, f32::from_bits(*value)));
        }
    }
    let _ = std::fs::write(scan.dir.join("scan.out"), reply);
}
