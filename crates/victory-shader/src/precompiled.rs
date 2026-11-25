//! Pipelines a title passes already compiled: GFX10 code in GPU memory and,
//! beside it, the hardware settings and a table of the descriptors the code
//! reads (device methods 0x1c8 and 0x218). The translator reads programs
//! from shader containers, so these are written into one: a `DXIL` part
//! whose `XDAT` index holds an `FSSM` blob, as the console compiler stores
//! its own programs (`crate::native`).
//!
//! The two methods are the title's cache of pipelines: when one fails, the
//! title makes the same pipeline from its shader containers. So this path
//! adds no pipeline, and is opt-in (`VICTORY_PRECOMPILED=1` in
//! `victory-gpu`). Only the compute form (0x218) is written.
//!
//! Layouts as observed (run of 2026-11-25 at 22bf30c, the first four calls
//! of each method):
//!
//! ```text
//! compute settings (0x1c B):
//!   +0x00 u32 x2 not known (0x4005cf00, 0x6c090102 in every call seen)
//!   +0x08 u32 0
//!   +0x0c u16 x2 thread-group size x, y; +0x10 u32 z
//!   +0x14 u32 COMPUTE_PGM_RSRC1; +0x18 u32 COMPUTE_PGM_RSRC2
//! binding table, compute (kind 0x403): u32 kind, u32 bytes after the
//!   header, u32 x3 (not known), then tagged entries
//! binding table, graphics stage (kind 2): u32 kind, u32 bytes after the
//!   header, u32 (not known), then tagged entries
//! entries: u32 (count << 16 | 0x440, the count not always that of 0x41),
//!   then
//!   0x41  32 B: tag, then the 20 bytes of an FSSM used-resource entry
//!         (u8 type, u8 root type, u16 0, u32 table offset in dwords, u32
//!         lower bound, u32 stride, u8 descriptor dwords, u8 first user-data
//!         dword, u8 root slot, u8 0), then u32 -1, u32 0
//!   0x42  32 B: tag, u32 0, u8 1, u8 range type (0 SRV, 1 UAV, 2 CBV),
//!         u8 (not known), u8 0, u32 register, u32 count, u32 x3 (0, -1, 0)
//!         (the bindings as declared; not needed here)
//!   0x43  20 B: tag, the 16-byte shader hash (also in the code's first
//!         instructions)
//!   0     end
//! ```
//!
//! The code starts with three `v_nop`s whose literals carry that hash, as
//! the console compiler's programs do.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{FourCC, u32_at};

/// What a binding table holds: the descriptors the code reads, as FSSM
/// used-resource entries (20 bytes each).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bindings {
    pub used: Vec<[u8; 20]>,
}

const TAG_USED: u32 = 0x41;
const TAG_DECLARED: u32 = 0x42;
const TAG_HASH: u32 = 0x43;
const KIND_COMPUTE: u32 = 0x403;
const KIND_GRAPHICS: u32 = 2;

/// Reads a binding table (see the module docs).
pub fn bindings(table: &[u8]) -> Result<Bindings, ShaderError> {
    let kind = u32_at(table, 0)?;
    let header = match kind {
        KIND_COMPUTE => 0x14,
        KIND_GRAPHICS => 0x0c,
        _ => return Err(Malformed("binding table of unknown kind")),
    };
    let size = u32_at(table, 4)? as usize;
    let end = header + size;
    if end > table.len() {
        return Err(Malformed("binding table longer than its part"));
    }
    // The count word, then the entries.
    let mut at = header + 4;
    let mut out = Bindings::default();
    while at + 4 <= end {
        match u32_at(table, at)? {
            TAG_USED => {
                let entry = table
                    .get(at + 4..at + 24)
                    .ok_or(Malformed("binding entry out of bounds"))?;
                out.used.push(entry.try_into().expect("20 bytes"));
                at += 32;
            }
            TAG_DECLARED => at += 32,
            TAG_HASH => at += 20,
            0 => break,
            _ => return Err(Malformed("binding table entry of unknown tag")),
        }
    }
    Ok(out)
}

/// The root parameters the code finds in user SGPRs, from the descriptors
/// it reads: one per root slot (a table's address, a root descriptor's,
/// or root constants), at the user-data dword the entries name. FSSM
/// layout entries, 20 bytes each.
fn layout(used: &[[u8; 20]]) -> Vec<[u8; 20]> {
    let mut out: Vec<[u8; 20]> = Vec::new();
    for u in used {
        let root = u32::from(u[1]);
        let (user_data, slot) = (u[17], u[18]);
        let sgprs = match root {
            // Table: the 32-bit heap address of the range.
            3 => 1,
            // Root descriptor: a 64-bit address.
            1 => 2,
            // Root constants: as many as the entry reads.
            0 => u32::from(u[16]).max(1),
            // Immediate samplers are in the code; heaps by index.
            _ => continue,
        };
        if out.iter().any(|e| e[17] == slot && e[0..4] == root.to_le_bytes()) {
            continue;
        }
        let mut e = [0u8; 20];
        e[0..4].copy_from_slice(&root.to_le_bytes());
        e[8..12].copy_from_slice(&sgprs.to_le_bytes());
        e[12..16].copy_from_slice(&u[8..12]);
        e[16] = user_data;
        e[17] = slot;
        out.push(e);
    }
    out
}

/// Whether the code runs as 64-lane waves: it writes the whole exec mask
/// with 64-bit operations, or (with no write to exec at all) by default,
/// as the console's compute programs seen do (one reads lane 36).
fn wave64(code: &[u8]) -> bool {
    use crate::gfx10::{Fields, Operand};
    let Ok(insts) = crate::gfx10::decode(code) else { return true };
    for i in &insts {
        let Fields::Sop { dst: Some(Operand::Scalar { code: 126, .. }), .. } = i.fields(true) else {
            continue;
        };
        let name = i.name().unwrap_or("");
        if name.ends_with("_b32") {
            return false;
        }
        if name.ends_with("_b64") {
            return true;
        }
    }
    true
}

// FSSM layout (`crate::native`).
const FSSM_VERSION: u32 = 0x101;
const SLOT_OFFSETS: [usize; 8] = [0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60];
const SLOT_CS: usize = 0x60;
const PAYLOAD_HEADER: usize = 0x68;
const RECORD_SIZE: usize = 0x54;
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const FLAG_WAVE32: u32 = 1 << 5;
const CS_STAGE_DATA: usize = 0x34;
const XBCH_VERSION: u32 = 3;
const KIND_NATIVE: u32 = 0x0d;

/// A container holding the compute program `code`, with its settings
/// (the 0x1c-byte part) and binding table, for the translator.
pub fn compute_container(settings: &[u8], code: &[u8], table: &[u8]) -> Result<Vec<u8>, ShaderError> {
    if settings.len() < 0x1c {
        return Err(Malformed("compute settings too short"));
    }
    let size = |at: usize| u32::from(u16::from_le_bytes([settings[at], settings[at + 1]]));
    let threads = [size(0x0c), size(0x0e), u32_at(settings, 0x10)?];
    if threads.contains(&0) || threads.iter().product::<u32>() > 1024 {
        return Err(Malformed("implausible thread-group size"));
    }
    let rsrc1 = u32_at(settings, 0x14)?;
    let rsrc2 = u32_at(settings, 0x18)?;
    let used = bindings(table)?.used;
    let wave64 = wave64(code);
    // COMPUTE_PGM_RSRC1: VGPRs in granules of 4 (wave64) or 8 (wave32),
    // SGPRs in granules of 8.
    let vgprs = ((rsrc1 & 0x3f) + 1) * if wave64 { 4 } else { 8 };
    let sgprs = ((rsrc1 >> 6 & 0xf) + 1) * 8;
    let mut stage = vec![0u8; CS_STAGE_DATA];
    stage[0x24..0x28].copy_from_slice(&rsrc2.to_le_bytes());
    for (k, t) in threads.iter().enumerate() {
        stage[0x28 + 4 * k..0x2c + 4 * k].copy_from_slice(&t.to_le_bytes());
    }
    let program = Program {
        code,
        wave64,
        vgprs: vgprs.min(256),
        sgprs: sgprs.min(106),
        lds: (rsrc2 >> 15 & 0x1ff) * 512,
        stage_data: &stage,
        used: &used,
    };
    Ok(container(&fssm(SLOT_CS, &program)))
}

struct Program<'a> {
    code: &'a [u8],
    wave64: bool,
    vgprs: u32,
    sgprs: u32,
    lds: u32,
    stage_data: &'a [u8],
    used: &'a [[u8; 20]],
}

/// An FSSM blob with one program, in the record slot at `slot`.
fn fssm(slot: usize, p: &Program) -> Vec<u8> {
    let layout = layout(p.used);
    let mut payload = vec![0u8; PAYLOAD_HEADER];
    payload[0..8].copy_from_slice(&6u64.to_le_bytes());
    for at in SLOT_OFFSETS {
        payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
    }
    let record = payload.len();
    payload[slot..slot + 8].copy_from_slice(&(record as u64).to_le_bytes());
    let layout_at = record + RECORD_SIZE + p.stage_data.len();
    let used_at = layout_at + 20 * layout.len();
    let code_at = used_at + 20 * p.used.len();
    let mut r = [0u8; RECORD_SIZE];
    r[0x00..0x08].copy_from_slice(&(code_at as u64).to_le_bytes());
    r[0x08..0x10].copy_from_slice(&(p.code.len() as u64).to_le_bytes());
    let flags = if p.wave64 { 0 } else { FLAG_WAVE32 };
    r[0x14..0x18].copy_from_slice(&flags.to_le_bytes());
    r[0x28..0x30].copy_from_slice(&(layout_at as u64).to_le_bytes());
    r[0x30..0x38].copy_from_slice(&(used_at as u64).to_le_bytes());
    r[0x38..0x3c].copy_from_slice(&(layout.len() as u32).to_le_bytes());
    r[0x3c..0x40].copy_from_slice(&(p.used.len() as u32).to_le_bytes());
    r[0x4c..0x50].copy_from_slice(&p.lds.to_le_bytes());
    r[0x50..0x54].copy_from_slice(&(p.vgprs | p.sgprs << 20).to_le_bytes());
    payload.extend_from_slice(&r);
    payload.extend_from_slice(p.stage_data);
    layout.iter().for_each(|e| payload.extend_from_slice(e));
    p.used.iter().for_each(|e| payload.extend_from_slice(e));
    payload.extend_from_slice(p.code);
    let mut out = b"FSSM".to_vec();
    out.extend_from_slice(&FSSM_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// A shader container whose `DXIL` part holds `fssm` (zlib-packed in an
/// `XBCH` blob behind an `XDAT` index, as the console compiler writes it).
fn container(fssm: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; zlib_rs::compress_bound(fssm.len())];
    let (stream, rc) = zlib_rs::compress_slice(&mut buf, fssm, zlib_rs::DeflateConfig::new(1));
    assert_eq!(rc, zlib_rs::ReturnCode::Ok, "deflating into a buffer of the bound");
    let mut blob = b"XBCH".to_vec();
    blob.extend_from_slice(&XBCH_VERSION.to_le_bytes());
    blob.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    blob.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    blob.extend_from_slice(stream);
    let mut part = vec![0u8; 0x18];
    part.extend_from_slice(b"XDAT");
    part.extend_from_slice(&1u32.to_le_bytes());
    part.extend_from_slice(&0x24u32.to_le_bytes());
    let offset = part.len() + 12;
    for v in [KIND_NATIVE, offset as u32, blob.len() as u32] {
        part.extend_from_slice(&v.to_le_bytes());
    }
    part.extend_from_slice(&blob);
    crate::dxbc::build(&[(FourCC::DXIL, &part)])
}

#[cfg(test)]
mod tests;
