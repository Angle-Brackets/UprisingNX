//! bytes that repeat a pattern of a few bytes over and over, as fills leave
//! buffers.

/// the pattern repeated over a block, a whole number of patterns of 2, 3 or 4
/// bytes, and of the vectors the CPU compares and stores at once.
const BLOCK: usize = 48;

/// the blocks looked at together, a few kilobytes, so that bytes differing
/// early stop the comparing early.
const GROUP: usize = 64;

fn block(pattern: &[u8]) -> [u8; BLOCK] {
    debug_assert!(!pattern.is_empty() && BLOCK.is_multiple_of(pattern.len()));
    std::array::from_fn(|i| pattern[i % pattern.len()])
}

/// the bytes compared and written at a time, the pattern over a few
/// kilobytes. the C library compares and copies them in fewer instructions
/// than a loop of 16-byte vectors.
const SPAN: usize = BLOCK * GROUP;

/// the pattern over a span.
fn span(pattern: &[u8]) -> [u8; SPAN] {
    let block = block(pattern);
    let mut span = [0u8; SPAN];
    for chunk in span.as_chunks_mut::<BLOCK>().0 {
        *chunk = block;
    }
    span
}

#[inline(always)]
fn holds_val_u64(u64s: &[u64], val: u64) -> bool {
    let chunks = u64s.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        let diff = (chunk[0] ^ val)
            | (chunk[1] ^ val)
            | (chunk[2] ^ val)
            | (chunk[3] ^ val)
            | (chunk[4] ^ val)
            | (chunk[5] ^ val)
            | (chunk[6] ^ val)
            | (chunk[7] ^ val);
        if diff != 0 {
            return false;
        }
    }
    for &w in rem {
        if w != val {
            return false;
        }
    }
    true
}

/// whether bytes hold the pattern over and over, from its first byte.
pub(crate) fn holds(bytes: &[u8], pattern: &[u8]) -> bool {
    if pattern.is_empty() {
        return true;
    }
    let p_len = pattern.len();
    if bytes.len() <= 16 {
        return bytes.iter().enumerate().all(|(i, &b)| b == pattern[i % p_len]);
    }
    if pattern.iter().all(|&b| b == 0) {
        let (prefix, u64s, suffix) = unsafe { bytes.align_to::<u64>() };
        return prefix.iter().all(|&b| b == 0)
            && holds_val_u64(u64s, 0)
            && suffix.iter().all(|&b| b == 0);
    }
    match p_len {
        1 => {
            let val = u64::from_ne_bytes([pattern[0]; 8]);
            let (prefix, u64s, suffix) = unsafe { bytes.align_to::<u64>() };
            prefix.iter().all(|&b| b == pattern[0])
                && holds_val_u64(u64s, val)
                && suffix.iter().all(|&b| b == pattern[0])
        }
        2 => {
            let (prefix, u64s, suffix) = unsafe { bytes.align_to::<u64>() };
            if prefix.len() % 2 == 0 {
                let h = u16::from_ne_bytes([pattern[0], pattern[1]]) as u64;
                let val = h | (h << 16) | (h << 32) | (h << 48);
                let p = [pattern[0], pattern[1]];
                prefix.iter().enumerate().all(|(i, &b)| b == p[i % 2])
                    && holds_val_u64(u64s, val)
                    && suffix.iter().enumerate().all(|(i, &b)| b == p[i % 2])
            } else {
                bytes.chunks(16).all(|c| c.iter().enumerate().all(|(i, &b)| b == pattern[i % 2]))
            }
        }
        3 => {
            let (prefix, u64s, suffix) = unsafe { bytes.align_to::<u64>() };
            let p = [pattern[0], pattern[1], pattern[2]];
            if !prefix.iter().enumerate().all(|(i, &b)| b == p[i % 3]) {
                return false;
            }
            let w = [
                u64::from_ne_bytes([p[0], p[1], p[2], p[0], p[1], p[2], p[0], p[1]]),
                u64::from_ne_bytes([p[1], p[2], p[0], p[1], p[2], p[0], p[1], p[2]]),
                u64::from_ne_bytes([p[2], p[0], p[1], p[2], p[0], p[1], p[2], p[0]]),
            ];
            let rot = prefix.len() % 3;
            let cw0 = w[rot];
            let cw1 = w[(rot + 2) % 3];
            let cw2 = w[(rot + 1) % 3];
            let chunks = u64s.chunks_exact(3);
            let rem = chunks.remainder();
            for chunk in chunks {
                if (chunk[0] ^ cw0) | (chunk[1] ^ cw1) | (chunk[2] ^ cw2) != 0 {
                    return false;
                }
            }
            let rem_expected = [cw0, cw1];
            for (i, &word) in rem.iter().enumerate() {
                if word != rem_expected[i] {
                    return false;
                }
            }
            let total_bytes_before_suffix = prefix.len() + u64s.len() * 8;
            suffix.iter().enumerate().all(|(i, &b)| b == p[(total_bytes_before_suffix + i) % 3])
        }
        4 => {
            let (prefix, u64s, suffix) = unsafe { bytes.align_to::<u64>() };
            if prefix.len() % 4 == 0 {
                let w = u32::from_ne_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]) as u64;
                let val = w | (w << 32);
                let p = [pattern[0], pattern[1], pattern[2], pattern[3]];
                prefix.iter().enumerate().all(|(i, &b)| b == p[i % 4])
                    && holds_val_u64(u64s, val)
                    && suffix.iter().enumerate().all(|(i, &b)| b == p[i % 4])
            } else {
                bytes.chunks(16).all(|c| c.iter().enumerate().all(|(i, &b)| b == pattern[i % 4]))
            }
        }
        _ => {
            let span = span(pattern);
            bytes.chunks(SPAN).all(|chunk| *chunk == span[..chunk.len()])
        }
    }
}

/// writes the pattern over bytes the way holds reads it, where they do not
/// hold it already. titles clear the same buffers to the same values frame
/// after frame, and reading memory costs less than writing it.
pub(crate) fn fill(bytes: &mut [u8], pattern: &[u8]) {
    if pattern.is_empty() {
        return;
    }
    if pattern.iter().all(|&b| b == 0) {
        bytes.fill(0);
        return;
    }
    let p_len = pattern.len();
    if bytes.len() <= 16 {
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = pattern[i % p_len];
        }
        return;
    }
    match p_len {
        1 => {
            if holds(bytes, pattern) {
                return;
            }
            bytes.fill(pattern[0]);
        }
        2 => {
            if holds(bytes, pattern) {
                return;
            }
            let (prefix, u64s, suffix) = unsafe { bytes.align_to_mut::<u64>() };
            if prefix.len() % 2 == 0 {
                let h = u16::from_ne_bytes([pattern[0], pattern[1]]) as u64;
                let val = h | (h << 16) | (h << 32) | (h << 48);
                let p = [pattern[0], pattern[1]];
                for (i, b) in prefix.iter_mut().enumerate() {
                    *b = p[i % 2];
                }
                u64s.fill(val);
                for (i, b) in suffix.iter_mut().enumerate() {
                    *b = p[i % 2];
                }
            } else {
                let span = span(pattern);
                for chunk in bytes.chunks_mut(SPAN) {
                    chunk.copy_from_slice(&span[..chunk.len()]);
                }
            }
        }
        3 => {
            if holds(bytes, pattern) {
                return;
            }
            let (prefix, u64s, suffix) = unsafe { bytes.align_to_mut::<u64>() };
            let p = [pattern[0], pattern[1], pattern[2]];
            for (i, b) in prefix.iter_mut().enumerate() {
                *b = p[i % 3];
            }
            let w = [
                u64::from_ne_bytes([p[0], p[1], p[2], p[0], p[1], p[2], p[0], p[1]]),
                u64::from_ne_bytes([p[1], p[2], p[0], p[1], p[2], p[0], p[1], p[2]]),
                u64::from_ne_bytes([p[2], p[0], p[1], p[2], p[0], p[1], p[2], p[0]]),
            ];
            let rot = prefix.len() % 3;
            let cw0 = w[rot];
            let cw1 = w[(rot + 2) % 3];
            let cw2 = w[(rot + 1) % 3];
            let mut chunks = u64s.chunks_exact_mut(3);
            for chunk in chunks.by_ref() {
                chunk[0] = cw0;
                chunk[1] = cw1;
                chunk[2] = cw2;
            }
            let rem = chunks.into_remainder();
            let rem_expected = [cw0, cw1];
            for (i, word) in rem.iter_mut().enumerate() {
                *word = rem_expected[i];
            }
            let total_bytes_before_suffix = prefix.len() + u64s.len() * 8;
            for (i, b) in suffix.iter_mut().enumerate() {
                *b = p[(total_bytes_before_suffix + i) % 3];
            }
        }
        4 => {
            if holds(bytes, pattern) {
                return;
            }
            let (prefix, u64s, suffix) = unsafe { bytes.align_to_mut::<u64>() };
            if prefix.len() % 4 == 0 {
                let w = u32::from_ne_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]) as u64;
                let val = w | (w << 32);
                let p = [pattern[0], pattern[1], pattern[2], pattern[3]];
                for (i, b) in prefix.iter_mut().enumerate() {
                    *b = p[i % 4];
                }
                u64s.fill(val);
                for (i, b) in suffix.iter_mut().enumerate() {
                    *b = p[i % 4];
                }
            } else {
                let span = span(pattern);
                for chunk in bytes.chunks_mut(SPAN) {
                    chunk.copy_from_slice(&span[..chunk.len()]);
                }
            }
        }
        _ => {
            let span = span(pattern);
            for chunk in bytes.chunks_mut(SPAN) {
                let span = &span[..chunk.len()];
                if *chunk != *span {
                    chunk.copy_from_slice(span);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the pattern over len bytes the plain way.
    fn repeated(pattern: &[u8], len: usize) -> Vec<u8> {
        (0..len).map(|i| pattern[i % pattern.len()]).collect()
    }

    /// a byte changed anywhere, in a group, in the tail or at either end of
    /// either, is seen, and filling leaves the pattern everywhere.
    #[test]
    fn a_byte_off_the_pattern_anywhere_is_seen_and_filled() {
        let lengths = [0, 1, 47, 48, 49, 48 * GROUP - 1, 48 * GROUP, 48 * GROUP + 1, 48 * GROUP * 3 + 29];
        for pattern in [&[0x00, 0x24, 0x24, 0x24][..], &[0x11, 0x22, 0x33], &[0xAB, 0xCD], &[0, 0, 0, 0]] {
            for len in lengths {
                let expected = repeated(pattern, len);
                assert!(holds(&expected, pattern), "{pattern:?} over {len}");
                let mut bytes = expected.clone();
                fill(&mut bytes, pattern);
                assert_eq!(bytes, expected);
                let places = [0, 1, len / 2, len.saturating_sub(48), len.saturating_sub(2), len.saturating_sub(1)];
                for at in places.into_iter().filter(|&at| at < len) {
                    let mut bytes = expected.clone();
                    bytes[at] ^= 0x40;
                    assert!(!holds(&bytes, pattern), "{pattern:?} over {len}, changed at {at}");
                    fill(&mut bytes, pattern);
                    assert_eq!(bytes, expected, "{pattern:?} over {len}, changed at {at}");
                }
                // what something else left there entirely
                let mut bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
                fill(&mut bytes, pattern);
                assert_eq!(bytes, expected);
            }
        }
    }

    /// the pattern starts at the first byte, bytes that start a byte into
    /// it hold it turned around.
    #[test]
    fn the_pattern_starts_at_the_first_byte() {
        let bytes = repeated(&[1, 2, 3, 4], 4096);
        assert!(holds(&bytes[4..], &[1, 2, 3, 4]));
        assert!(!holds(&bytes[1..], &[1, 2, 3, 4]));
        assert!(holds(&bytes[1..], &[2, 3, 4, 1]));
        assert!(!holds(&bytes, &[1, 2]));
    }
}
