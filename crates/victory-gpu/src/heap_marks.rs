//! Which buffer views of the view heap shaders store through when they pick
//! them at an index they compute (SM 6.6 dynamic resources).
//!
//! On the console such a view names title memory itself: the store is in
//! the title's memory at once. Here it lands in the host copy the view
//! shows, and only ranges a list declared written
//! ([`crate::shadow::Shadow::gpu_writes`]) are compared and copied back
//! after its submission. Tables and root views say which buffers a draw can
//! write; a computed index says nothing, and declaring every unordered
//! access view of the heap is not affordable (960 MB of them in the world,
//! for some 30 lists a frame: run of 2026-11-24 on main d3decdc).
//!
//! So the translated shader says it: before every store or atomic operation
//! through a picked buffer view it sets dword `slot` of one buffer of ours
//! (`victory_shader::emit::SLOT_HEAP_MARKS`, bound as a root unordered
//! access view). The marks are read and cleared here once a frame; a slot
//! marked within the last [`KEEP_FRAMES`] frames is one shaders write, and
//! lists that run a shader storing through picked views declare the ranges
//! of those slots' views written, and nothing else.
//!
//! shortcut: the marks come back after the shader ran, so the stores of the
//! frame a slot is first marked in (and of the frames until the mark is
//! read) are not copied back; a buffer a shader writes once and never again
//! stays unseen by the title. [`FIRST_MARKS`] counts them. The whole answer
//! needs the copy-back to look at the marks after the submission, with
//! snapshots taken for every unordered view beforehand.
//! shortcut: one buffer of marks for every view heap; slot `n` marked in
//! one heap is taken as marked in the others (more declared, never less).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::HEAP_MARKS_DWORDS;
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

/// Frames a slot stays one that shaders write after its last mark: a
/// buffer written every few frames is not dropped and found again (losing
/// a frame's stores each time).
pub(crate) const KEEP_FRAMES: u64 = 120;

/// Slots found marked for the first time (or again after [`KEEP_FRAMES`]
/// without a mark), since the start: each is at least a frame of stores
/// that were not copied back.
pub(crate) static FIRST_MARKS: AtomicU64 = AtomicU64::new(0);
/// Marks found on slots whose view is not an unordered access one, since
/// the start: the host made a read-only view where a shader stores, and
/// the store was dropped.
pub(crate) static READ_ONLY_MARKS: AtomicU64 = AtomicU64::new(0);
/// Slots taken as written at the last frame's reading.
pub(crate) static WRITTEN_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Whether what shaders store through picked buffer views goes back into
/// title memory (`VICTORY_HEAP_WRITES=0`: no; the marks are still read and
/// counted).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_WRITES").is_none_or(|v| v != "0"))
}

/// Per heap slot: the frame its mark was last found set.
#[derive(Default)]
struct Written(HashMap<u32, u64>);

impl Written {
    /// `slot` was found marked in `frame`. Whether it was not known as
    /// written before.
    fn mark(&mut self, slot: u32, frame: u64) -> bool {
        match self.0.insert(slot, frame) {
            Some(last) => frame.saturating_sub(last) > KEEP_FRAMES,
            None => true,
        }
    }

    /// Forgets the slots not marked for [`KEEP_FRAMES`] frames.
    fn expire(&mut self, frame: u64) {
        self.0.retain(|_, last| frame.saturating_sub(*last) <= KEEP_FRAMES);
    }

    fn has(&self, slot: u32) -> bool {
        self.0.contains_key(&slot)
    }
}

/// `ranges` (`[from, to)` each) as the fewest ranges covering the same
/// bytes, ascending.
pub(crate) fn merged(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

struct Marks {
    buffer: ID3D12Resource,
    data: *mut u32,
    gpu: u64,
    written: Written,
}

// SAFETY: the mapping is of a buffer the struct owns; all access is under
// the mutex.
unsafe impl Send for Marks {}

static MARKS: Mutex<Option<Marks>> = Mutex::new(None);

/// The GPU address of the marks for a draw of `list` to bind as its root
/// unordered access view; the list holds the buffer. None if the host
/// could not make it.
pub(crate) fn bind(list: &HostList) -> Option<u64> {
    let mut guard = MARKS.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let bytes = u64::from(HEAP_MARKS_DWORDS) * 4;
        let (buffer, data) = host()?.shared_buffer(bytes)?;
        crate::host::name(&buffer, || "marks of heap slots shaders store through".to_string());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::info!("a shader stores through buffer views it picks from the view heap by index: the slots it marks are copied back");
        *guard = Some(Marks {
            buffer,
            data: data.cast(),
            gpu,
            written: Written::default(),
        });
    }
    let m = guard.as_ref()?;
    const KEEP: usize = usize::MAX - 0x60;
    if list.first_use(KEEP) {
        list.keep(&m.buffer);
    }
    Some(m.gpu)
}

/// Once a frame: reads and clears the marks of `slots` (each a heap slot
/// holding a buffer view, and whether the view is an unordered access one)
/// and returns those of them shaders write, by the marks of this and
/// earlier frames. Empty until a shader that marks has been bound.
pub(crate) fn take(frame: u64, slots: impl Iterator<Item = (u32, bool)>) -> Vec<u32> {
    let mut guard = MARKS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return Vec::new();
    };
    m.written.expire(frame);
    let mut out = Vec::new();
    for (slot, unordered) in slots {
        // The last dword is the sink of lanes that do not run.
        if slot >= HEAP_MARKS_DWORDS - 1 {
            continue;
        }
        // SAFETY: inside the mapping; the GPU may be writing, and a mark
        // missed now is set again by the next store.
        let at = unsafe { m.data.add(slot as usize) };
        // SAFETY: as above.
        if unsafe { at.read_volatile() } != 0 {
            // SAFETY: as above.
            unsafe { at.write_volatile(0) };
            if !unordered {
                READ_ONLY_MARKS.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if m.written.mark(slot, frame) {
                FIRST_MARKS.fetch_add(1, Ordering::Relaxed);
            }
        }
        if unordered && m.written.has(slot) {
            out.push(slot);
        }
    }
    WRITTEN_SLOTS.store(out.len() as u64, Ordering::Relaxed);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_where_they_touch_or_overlap() {
        assert_eq!(merged(vec![]), vec![]);
        assert_eq!(merged(vec![(8, 8), (4, 2)]), vec![]);
        assert_eq!(
            merged(vec![(40, 50), (0, 10), (10, 20), (15, 18), (19, 30), (60, 70), (45, 61)]),
            vec![(0, 30), (40, 70)]
        );
    }

    #[test]
    fn a_slot_stays_written_for_the_kept_frames_after_its_last_mark() {
        let mut w = Written::default();
        assert!(w.mark(7, 100));
        assert!(!w.mark(7, 101));
        w.expire(101 + KEEP_FRAMES);
        assert!(w.has(7));
        w.expire(102 + KEEP_FRAMES);
        assert!(!w.has(7));
        // Found again: a first mark once more.
        assert!(w.mark(7, 400));
        assert!(!w.has(8));
    }
}
