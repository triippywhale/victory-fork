//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! **While the structure builder is off** (`pipeline::builder_live`) the
//! structures are not built, and what lies at their addresses is whatever
//! the title's memory held before. The programs that walk them get
//! [`empty`] then: a directory without entries, so every read by address
//! gives 0 and every store is dropped. A structure of zeros is the empty
//! scene (its root is a triangle of three equal points, which no ray
//! hits), so every ray misses: nothing is occluded, the title shades as it
//! does while its structures are still streaming in. Nothing is copied
//! for it either (no copy is numbered, brought up to date or compared).
//! `VICTORY_RT_WALK_UNBUILT=1` has them read title memory as before.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (windows of 1 to 4 MiB mostly).
//! Every list with such a shader brings all numbered copies up to date
//! and, if one of its shaders can store by address
//! (`pipeline::Translated::stores_by_address`), declares written (the bytes
//! are compared on the CPU afterwards, as for any unordered access view)
//! only the copies shaders write (fxceless #346): stores and atomics set a
//! mark per copy (`victory_shader::emit::MEMORY_MARKS`), read once a frame,
//! and a list declares the copies marked in the last [`MARK_FRAMES`]
//! frames, those its dispatches name and those numbered while it is
//! recorded. Declaring every copy cost 3.9 GB of snapshots and compares a
//! frame with the builder on (run 3 at 266cd04); `VICTORY_MEMORY_MARKS=0`
//! does so again.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_MARKS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer, taken
/// from each view heap's dynamic slots for the heap's life). A listed frame
/// of the opening cutscene names about 50 windows; with the structure
/// builder on, 1,023 were all numbered two seconds after it started (run
/// main-8918d34-builder, windows on the 1 MiB grid), and every page that
/// found no number after that read zeros for the rest of the run; 4,095
/// were all numbered a minute in (run 3 at 266cd04: walls missing). 8,191
/// as in fxceless #980.
const COPIES: u32 = 8191;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// A copy counts as one shaders write for this many frames after a mark:
/// marks are read once a frame, while lists recorded earlier still run.
const MARK_FRAMES: u64 = 8;
const _: () = assert!(COPIES < MEMORY_MARKS);

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Per copy: the frame its mark was last found set in (`u64::MAX`:
    /// never), and the frame a list last declared it written in.
    marked: Vec<(u64, u64)>,
    /// For [`empty`]: a directory without entries and the fault notes
    /// nothing reads, made at its first use.
    unbuilt: Option<[(ID3D12Resource, *mut u32, u64); 2]>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

/// A buffer of `dwords` zero dwords the CPU and the card share: the
/// buffer, its mapping, its GPU address.
fn zeroed(dwords: u64, what: &str) -> Option<(ID3D12Resource, *mut u32, u64)> {
    let (buffer, data) = host()?.shared_buffer(dwords * 4)?;
    crate::host::name(&buffer, || format!("title memory by address: {what}"));
    // SAFETY: a fresh mapping of at least that many bytes.
    unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some((buffer, data.cast::<u32>(), gpu))
}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: zeroed(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: zeroed(u64::from(MEMORY_FAULTS + MEMORY_MARKS), "faults and marks")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            marked: Vec::new(),
            unbuilt: None,
        })
    }

    /// `heap`'s run of slots for the copies' views (its first slot and how
    /// many copies have their view there), taken at its first use.
    fn run(&mut self, heap: &HostHeap) -> Option<(u32, u32)> {
        let key = std::ptr::from_ref(heap) as usize;
        if let Some(&run) = self.heaps.get(&key) {
            return Some(run);
        }
        // Held for the heap's life: not given back with a list's slots.
        let base = heap.dynamic(COPIES + 1)?;
        heap.null_raw_uav_at(base);
        self.heaps.insert(key, (base, 0));
        Some((base, 0))
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's. Returns the index in `copies` of the page's copy.
    fn map(&mut self, address: u64) -> Option<usize> {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return None;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        match unsafe { entry.read() } {
            0 => {}
            n => return Some(n as usize - 1),
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return None;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return None;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        COPIES,
                        address = format_args!("{start:#x}"),
                        "title memory by address: no number left for another copy; its pages read zeros"
                    )
                });
                return None;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
        Some(n)
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// The marks set since they were last read (fxceless #346): their
    /// copies are ones shaders write, as of `frame`.
    fn take_marks(&mut self, frame: u64) {
        self.marked.resize(self.copies.len(), (u64::MAX, u64::MAX));
        // Marks of copies no list had declared written lately: stores that
        // were not copied back.
        let mut late = 0;
        for n in 0..self.copies.len() {
            // SAFETY: inside the mapping, past the fault notes; the GPU may
            // be writing: a mark set after this read is found next frame.
            let at = unsafe { self.faults.1.add(MEMORY_FAULTS as usize + n + 1) };
            // SAFETY: as above.
            if unsafe { at.read_volatile() } == 0 {
                continue;
            }
            // SAFETY: as above.
            unsafe { at.write_volatile(0) };
            let (_, declared) = self.marked[n];
            if declared == u64::MAX || frame.wrapping_sub(declared) > MARK_FRAMES {
                late += 1;
            }
            self.marked[n].0 = frame;
        }
        if late != 0 {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::info!(
                    copies = late,
                    of = self.copies.len(),
                    "title memory by address: shaders stored to copies no list had declared written (those stores were not copied back); they are declared from now on"
                );
            }
        }
    }

    /// Whether copy `n` is one shaders wrote lately.
    fn written_lately(&self, n: usize, frame: u64) -> bool {
        self.marked.get(n).is_some_and(|&(marked, _)| {
            marked != u64::MAX && frame.wrapping_sub(marked) <= MARK_FRAMES
        })
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it. Their indices go to `named`.
    fn name(&mut self, address: u64, named: &mut Vec<usize>) {
        named.extend(self.map(address));
        named.extend(self.map((address & !(CELL - 1)) + CELL));
    }
}

/// Bytes of a block of list upload memory looked through for title
/// addresses, from a root SRV's address on: some 340 geometry descriptions.
const SCAN: u64 = 16 << 10;

/// The 8-byte values of `bytes` (from an 8-byte boundary) that are title
/// addresses as the builder writes them: bits 32 to 39 set, nothing above,
/// and a low dword of 64 KiB or more (as for an address in two constants;
/// an address that is not the title's memory is refused when named).
fn title_addresses(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes.chunks_exact(8).filter_map(|q| {
        let v = u64::from_le_bytes(q.try_into().ok()?);
        ((1..=0xFF).contains(&(v >> 32)) && v as u32 >= 0x1_0000).then_some(v)
    })
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    records: &[u64],
    writes: bool,
    builder: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
        m.take_marks(frame);
    }
    let numbered = m.copies.len();
    let mut named = Vec::new();
    for &address in addresses {
        m.name(address, &mut named);
    }
    // List upload memory (`crate::list::upload`, there only with the
    // title's builder asked for): the builder reaches it through pointers
    // it reads out of its root views. A chunk is a window; once numbered
    // it stays so (fxceless 2d95091e).
    #[cfg(windows)]
    if let Some((start, len)) = crate::list::upload::span() {
        for chunk in (start..start + len).step_by(crate::list::upload::CHUNK) {
            let _ = m.map(chunk);
        }
    }
    // Title addresses in the blocks of list upload memory a dispatch reads
    // through its root SRVs (fxceless 273fc655): the builder's leaf
    // rehydration takes its geometry (index and vertex buffers) by address
    // out of its 48-byte records there, which no root argument names; the
    // ones in no copy when the stage ran read zeros (leaves without
    // triangles). The builder fills the records before it records the
    // dispatch. Looked through up to what the list has handed out,
    // [`SCAN`] bytes at most.
    #[cfg(windows)]
    for &address in records {
        let Some(handed) = crate::list::upload::handed_end(address) else {
            continue;
        };
        let from = address & !7;
        let end = handed.min(address.saturating_add(SCAN));
        if end <= from {
            continue;
        }
        let mut bytes = vec![0u8; (end - from) as usize];
        if crate::host::read_into(from as *const u8, &mut bytes) {
            for pointer in title_addresses(&bytes) {
                m.name(pointer, &mut named);
            }
        }
    }
    #[cfg(not(windows))]
    let _ = records;
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    // The low dword of a pair is not taken in 256-byte units as well: of
    // the windows the builder's dispatches named that way, 717 of 802 in
    // its first frame were such low dwords (fxceless 2d95091e).
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32, &mut named);
        } else if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8, &mut named);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = m.run(heap)?;
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Bring up to date (and, with `writes`, declare for write-back) every
    // numbered copy this list has not covered yet, including copies numbered
    // by another command list since this list's previous dispatch.
    const KEEP: usize = usize::MAX - 0x6d;
    let (read_from, write_from) = list.memory_sync_range(m.copies.len(), writes);
    for shadow in &m.copies[read_from..] {
        let (start, len) = shadow.span();
        shadow.refresh(list, start, len);
    }
    if writes {
        // Written (fxceless #346): of the copies this list has not looked
        // at yet, those shaders marked lately and those numbered now; and
        // the copies this dispatch names. Each once per list.
        // shortcut: a copy this list passed over that shaders mark later
        // in the frame is not declared by this list; its stores there are
        // not copied back (the log counts such marks).
        const WRITTEN: usize = usize::MAX - 0x1000;
        let all = !marks();
        m.marked.resize(m.copies.len(), (u64::MAX, u64::MAX));
        let fresh = write_from..m.copies.len();
        let wanted: Vec<usize> = fresh
            .clone()
            .filter(|&n| all || n >= numbered || m.written_lately(n, frame))
            .chain(named.iter().copied().filter(|n| !fresh.contains(n)))
            .collect();
        for n in wanted {
            if !list.first_use(WRITTEN - n) {
                continue;
            }
            let shadow = &m.copies[n];
            let (start, len) = shadow.span();
            // Bounded for the structure builder (`crate::pipeline::bvh_bound`).
            // shortcut: a list's copies are declared once, by whichever
            // program that can store comes first; a list mixing the builder
            // with another such program follows the first.
            crate::pipeline::builder_writes(list, shadow, start, len, builder);
            m.marked[n].1 = frame;
        }
    }
    if list.first_use(KEEP) {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

/// Whether only the copies shaders mark are taken for written (on by
/// default; `VICTORY_MEMORY_MARKS=0` takes every copy, as before: each
/// list then snapshots and compares all of them, 3.9 GB a frame with the
/// structure builder on).
fn marks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_MARKS").is_none_or(|v| v != "0"))
}

/// Evicts cached descriptor-run state for `heap` when its owning `Heap` is
/// dropped so a later `Heap` allocated at the same address cannot reuse stale
/// `(base, made)` slot offsets.
pub(crate) fn forget_heap(heap: &HostHeap) {
    let key = std::ptr::from_ref(heap) as usize;
    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.heaps.remove(&key);
    }
}

/// `VICTORY_RT_WALK_UNBUILT=1`: the programs that walk the ray-tracing
/// structures read title memory while the structure builder is off too
/// (what they did before [`empty`]; for comparing runs).
pub(crate) fn walks_unbuilt() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RT_WALK_UNBUILT").is_some_and(|v| v == "1"))
}

/// Dispatches and draws given [`empty`] so far.
static EMPTY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// For a draw whose shader walks ray-tracing structures while the
/// structure builder is off (see the module): title memory by address
/// with no page in it. Every read gives 0 and every store is dropped;
/// nothing is copied, and the pages the shader asks for are not noted.
/// None if the host cannot hold it; the draw is then dropped.
pub(crate) fn empty(list: &HostList, heap: &HostHeap) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    // The run's first slot is the null view, which is all a directory
    // without entries leads to.
    let (base, _) = m.run(heap)?;
    if m.unbuilt.is_none() {
        m.unbuilt = Some([
            zeroed(
                u64::from(MEMORY_PAGES) * 2,
                "directory of unbuilt structures",
            )?,
            zeroed(
                u64::from(MEMORY_FAULTS + MEMORY_MARKS),
                "faults of unbuilt structures",
            )?,
        ]);
    }
    let [directory, faults] = m.unbuilt.as_ref()?;
    const UNBUILT: usize = usize::MAX - 0x6f;
    if list.first_use(UNBUILT) {
        list.keep(&directory.0);
        list.keep(&faults.0);
    }
    let n = EMPTY.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n == 1 || n.is_multiple_of(4096) {
        tracing::info!(
            so_far = n,
            frame = crate::frame::current(),
            "ray tracing: the structure builder is off, so the programs that walk its structures are given empty ones (every ray misses) and no title memory is copied for them"
        );
    }
    Some(Ready {
        directory: directory.2,
        faults: faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{HeapDesc, create_heap, heap};
    use crate::object::{S_OK, This, guid};

    #[test]
    fn title_addresses_are_found_in_geometry_records() {
        // A geometry description as the builder copies it into upload memory
        // (48 bytes): no transform, index and vertex formats, counts, the
        // index buffer, the vertex buffer and its stride.
        let mut record = Vec::new();
        for v in [
            0u64,
            0x10_0000_0035,
            0x508_0000_1263,
            0xb_13f6_53a0,
            0xb_13f0_8bc0,
            8,
        ] {
            record.extend_from_slice(&v.to_le_bytes());
        }
        let found: Vec<u64> = title_addresses(&record).collect();
        assert_eq!(found, [0xb_13f6_53a0, 0xb_13f0_8bc0]);
    }

    #[test]
    fn dropping_a_view_heap_evicts_its_memory_heaps_entry() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &desc, &iid, &mut h), S_OK);
        let list = host.new_list().unwrap();
        let host_heap = heap(h).host().expect("host view heap");
        let key = std::ptr::from_ref(host_heap) as usize;

        assert!(ready(&list, host_heap, &[], &[], &[], false, false).is_some());
        {
            let guard = MEMORY.lock().unwrap();
            assert!(guard.as_ref().unwrap().heaps.contains_key(&key));
        }

        // Release the title heap (COM vtable slot 2): `Heap::drop` must evict `key`.
        // SAFETY: `h` is a live COM object from `create_heap`.
        let release: extern "system" fn(This) -> u32 =
            unsafe { std::mem::transmute(*(*(h as *const *const *const ())).add(2)) };
        assert_eq!(release(h), 0);

        let guard = MEMORY.lock().unwrap();
        assert!(
            !guard.as_ref().unwrap().heaps.contains_key(&key),
            "MEMORY.heaps retained dropped HostHeap key {key:#x}"
        );
    }

    use victory_shader::bvh::{MISS, NONE, Ray, intersect};

    /// What [`super::empty`] rests on: in memory that reads as zeros no
    /// ray hits anything, and no pointer leads anywhere but back to the
    /// node of zeros.
    #[test]
    fn a_structure_of_zeros_is_missed_by_every_ray() {
        let zeros = [0u32; 28];
        let rays = [
            Ray::new([0.0; 3], [0.0, 0.0, 1.0], 100.0),
            Ray::new([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], 1e30),
            Ray::new([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0),
            Ray::new([3.0, -4.0, 0.5], [0.3, 0.9, -0.1], f32::MAX),
        ];
        for ray in &rays {
            for sort in [false, true] {
                // A triangle of three equal points (the node a zero
                // pointer names), an instance or a procedural node.
                for kind in [0, 1, 2, 3, 6, 7] {
                    assert_eq!(
                        intersect(&zeros, kind, ray, sort),
                        MISS.map(f32::to_bits),
                        "kind {kind}"
                    );
                }
                // A box node of zeros: its children are the zero pointer,
                // that triangle again.
                for kind in [4, 5] {
                    for child in intersect(&zeros, kind, ray, sort) {
                        assert!(child == NONE || child == 0, "kind {kind}: {child:#x}");
                    }
                }
            }
        }
    }
}
