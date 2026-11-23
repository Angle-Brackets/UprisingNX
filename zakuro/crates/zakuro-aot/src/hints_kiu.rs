//! Built-in indirect-branch control-flow hints (pure integer word deltas, zero
//! ROM code) for Kid Icarus: Uprising (USA, 0004000000030100), plus optional
//! external `.hints` text parsing.

pub const KIU_USA_PROGRAM_ID: u64 = 0x0004_0000_0003_0100;
pub const KIU_USA_TEXT_HASH: u64 = 0xD6E0_28C6_175A_2B07;
pub const KIU_USA_TEXT_BASE: u32 = 0x0010_0000;

static KIU_USA_DELTAS: &[u8] = include_bytes!("hints_kiu_usa.bin");

/// Decodes the built-in hint addresses for `(program_id, text_hash)`, if known.
pub fn builtin_hints(program_id: u64, text_hash: u64) -> Vec<u32> {
    if program_id != KIU_USA_PROGRAM_ID || text_hash != KIU_USA_TEXT_HASH {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(37_100);
    let mut word = KIU_USA_TEXT_BASE / 4;
    let mut i = 0;
    while i < KIU_USA_DELTAS.len() {
        let mut delta = 0u32;
        let mut shift = 0u32;
        loop {
            let b = KIU_USA_DELTAS[i];
            i += 1;
            delta |= ((b & 0x7F) as u32) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        word = word.wrapping_add(delta);
        out.push(word * 4);
    }
    out
}

/// Parses hexadecimal hint addresses from a `.hints` text file.
pub fn parse_hints_text(text: &str) -> Vec<u32> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('='))
        .filter_map(|l| u32::from_str_radix(l.trim_start_matches("0x"), 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_kiu_usa_hints() {
        let hints = builtin_hints(KIU_USA_PROGRAM_ID, KIU_USA_TEXT_HASH);
        assert_eq!(hints.len(), 37_037);
        assert_eq!(hints.first().copied(), Some(0x0010_004C));
        assert_eq!(hints.last().copied(), Some(0x00C9_3804));
        assert!(hints.windows(2).all(|w| w[0] < w[1] && w[0] & 3 == 0));
    }
}
