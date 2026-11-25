//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_FAILED, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((now_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Host `(available physical RAM, available commit charge)` in bytes, from
/// `GlobalMemoryStatusEx`.
fn memory_status() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some((m.ullAvailPhys, m.ullAvailPageFile))
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    memory_status().map(|(phys, _)| phys)
}

/// Remaining Windows commit charge (`ullAvailPageFile`) in bytes.
pub(crate) fn available_commit_ram() -> Option<u64> {
    memory_status().map(|(_, commit)| commit)
}

/// What staging and shadow buffers may use: the lower of available physical
/// RAM and commit headroom, since both run out (fxceless #822).
pub(crate) fn available_cache_ram() -> Option<u64> {
    memory_status().map(|(phys, commit)| phys.min(commit))
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Drops what waited for submissions up to `finished`.
fn bury(finished: u64) {
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired)
            .into_iter()
            .partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Time detiles waited to reuse a staging buffer a submission sent ahead
/// still copies out of (`Host::wait_for_staging`), since the last line.
static STAGING_WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Upload buffers kept for the next ask of their size instead of let go
/// ([`Host::upload_pooled`], [`give_upload`]): the buffer, its mapping and
/// its bytes, oldest first, and the bytes in all. The run of main ce79bfb
/// made 195,898 upload buffers (69 GB) in seven minutes, most of them the
/// staging of a texture let go and made again, each a committed resource
/// the driver reserves, commits and zeroes, with under 4 GB left for the
/// PC to commit.
struct UploadPool {
    kept: std::collections::VecDeque<(ID3D12Resource, usize, u64)>,
    bytes: u64,
}

static UPLOAD_POOL: Mutex<UploadPool> = Mutex::new(UploadPool {
    kept: std::collections::VecDeque::new(),
    bytes: 0,
});

/// Upload buffers taken from [`UPLOAD_POOL`] instead of made, and their
/// bytes, since the start (for the memory line).
pub(crate) static UPLOADS_TAKEN: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The most [`UPLOAD_POOL`] keeps, in bytes: `VICTORY_UPLOAD_POOL_MB`
/// (256 by default; 0 keeps none, and every upload buffer is made anew as
/// before).
fn upload_pool_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_UPLOAD_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(256)
            .min(4096)
            << 20
    })
}

/// The bytes an upload buffer for `size` bytes is made with, so that asks
/// of nearly the same size share buffers: a multiple of 64 KiB up to
/// 1 MiB, of an eighth of the next power of two above.
pub(crate) fn upload_room(size: u64) -> u64 {
    let size = size.max(1);
    let step = if size <= 1 << 20 { 64 << 10 } else { size.next_power_of_two() >> 3 };
    size.next_multiple_of(step)
}

/// Gives back an upload buffer of [`Host::upload_pooled`] that nothing on
/// the card reads any more (every submission that named it has finished):
/// kept for the next ask of its size, the oldest ones let go beyond
/// [`upload_pool_most`].
pub(crate) fn give_upload(buffer: ID3D12Resource, data: *mut u8, room: u64) {
    let most = upload_pool_most();
    if room > most {
        return;
    }
    let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.kept.push_back((buffer, data as usize, room));
    pool.bytes += room;
    while pool.bytes > most {
        let Some((_, _, gone)) = pool.kept.pop_front() else {
            break;
        };
        pool.bytes -= gone;
    }
}

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// Per-list D3D12 state of `Shadow` buffers. Every `Shadow` buffer is created
/// in `D3D12_RESOURCE_STATE_COMMON` and decays back to `COMMON` at the end of
/// `ExecuteCommandLists`, so state tracking is strictly local to each list.
#[derive(Default)]
struct BufferStates {
    /// Raw `ID3D12Resource` pointer -> `(current state, live handle)`.
    /// Holding a clone keeps the COM address unique until `HostList::reset`.
    entries: std::collections::HashMap<usize, (D3D12_RESOURCE_STATES, ID3D12Resource)>,
}

impl BufferStates {
    fn transition_for(
        &mut self,
        buffer: &ID3D12Resource,
        wanted: D3D12_RESOURCE_STATES,
    ) -> Option<D3D12_RESOURCE_STATES> {
        let key = buffer.as_raw() as usize;
        match self.entries.entry(key) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, buffer.clone()));
                (wanted != D3D12_RESOURCE_STATE_COMMON).then_some(D3D12_RESOURCE_STATE_COMMON)
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().0, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// A host command list and its allocator, always used together.
/// The bytes of one buffer of hoisted clamped indirect arguments.
const HOIST_CHUNK: u64 = 256 << 10;

/// Where [`HostList::hoist_mark`]s come from.
static HOIST_MARKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A list's buffers for hoisted clamped indirect arguments
/// ([`HostList::clamp_room`]): the buffers, how many this recording uses,
/// the bytes taken of the last of those. Each starts a recording in the
/// common state (buffers decay to it when a submission has run).
#[derive(Default)]
pub(crate) struct Hoisted {
    chunks: Vec<ID3D12Resource>,
    used: usize,
    at: u64,
}

pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Per-list D3D12 state of `Shadow` buffers (`COMMON` at list start,
    /// decaying back to `COMMON` when `ExecuteCommandLists` finishes).
    buffer_states: Mutex<BufferStates>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// When this list holds the hoisted clamps of the list after it
    /// (`crate::indirect::hoisting`): the buffers the clamped arguments go
    /// to ([`HostList::clamp_room`]).
    hoisted: Mutex<Hoisted>,
    /// Changes whenever something recorded may change indirect arguments
    /// (a title copy, a shader write to be copied back, deferred work):
    /// clamps hoisted before the list hold while it stays the same
    /// ([`HostList::hoist_mark`]). Taken from [`HOIST_MARKS`], so no two
    /// lists, and no list before and after a reset, share a value.
    hoist_mark: std::sync::atomic::AtomicU64,
    /// With [`fill_on_card`]: a device-local buffer constant buffer views of
    /// the list read, filled by copies on the card ([`HostList::card_constants`]):
    /// the buffer, the bytes used, and whether it is in the constant buffer
    /// state (it decays to the common state when a submission has run).
    card_constants: Mutex<Option<(ID3D12Resource, u64, bool)>>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping and its host GPU address.
struct Chunk {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        _buffer: buffer,
        data: data as usize,
        gpu,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, SPARE_MOST)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// With [`relay`]: the push of what shaders wrote ([`CARD_PUSH`]), if
    /// it could be made and [`card_push`] is on.
    push: Option<CardPush>,
}

/// What [`CARD_PUSH`] needs: its pipeline, the slots in PC memory the
/// parts go down to, the bookkeeping beside them, and where the thread
/// that lands flights has read up to.
struct CardPush {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// [`PUSH_SLOTS`] slots of [`PUSH_SLOT`] bytes: PC memory, mapped
    /// (buffer, mapping).
    slots: (ID3D12Resource, usize),
    /// Dword 0: slots taken so far (wrapping); dword 1: slots the landing
    /// thread has read ([`CardPush::cursor`], told to the card); from byte
    /// [`PUSH_COUNTS`] a dword per plan tag (modulo [`PUSH_TAGS`]): the
    /// slots its pass took; from byte [`PUSH_RECORDS`] two dwords per slot:
    /// the tag of the plan that filled it and `dword << 4 | part` (the
    /// piece's dword in `differs` and the part). PC memory, mapped.
    meta: (ID3D12Resource, usize),
    /// The next plan's tag.
    next_tag: u32,
    /// The first slot no landed plan took.
    cursor: u32,
}

/// Slots of [`CardPush`] (a power of two).
const PUSH_SLOTS: u32 = 1 << 15;
/// Bytes of a slot: a 4 KiB part from the dword at or below its first
/// byte, and the bytes up to 3 that lie past it.
const PUSH_SLOT: u32 = 4112;
/// Plan tags whose counts are kept at once (more than plans in the air).
const PUSH_TAGS: u32 = 1024;
/// Where [`CardPush`]'s counts and records start in its meta buffer.
const PUSH_COUNTS: u32 = 64;
const PUSH_RECORDS: u32 = PUSH_COUNTS + 4 * PUSH_TAGS;

/// Whether what shaders wrote to copies in the card's memory goes down to
/// PC memory in the submission that wrote it (`VICTORY_CARD_PUSH=0` turns
/// it off; with [`relay`] only).
///
/// The compare after a submission's lists flagged the 4 KiB parts that
/// changed, and the thread that lands flights then sent a list of its own
/// to fetch them and waited for it: that fetch runs behind everything sent
/// since, so the wait was for the card's later work too (run C, main
/// d3decdc: 10.5 fetches and 45 ms a frame, 68 MB). Now a pass after the
/// compare, in the same list, copies each flagged part into a slot of a
/// ring in PC memory ([`CARD_PUSH`]), and the landing merges from there
/// with no fetch and no wait of its own. A part that found no free slot is
/// fetched as before.
fn card_push() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_PUSH").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, of [`card_push`]: parts merged from the
/// slots, their bytes, parts fetched because they were not in a slot.
/// Then parts left to a later plan of the same landing that pushed them too,
/// the runs of touching parts those merges took (one title memory read and
/// write each), and the runs that put nothing into title memory (flagged,
/// but what the card held there was what the CPU's side held). Then the
/// copies those runs were of (summed over landings) and the microseconds
/// the merges took ([`merge_pushed`], wall time).
static PUSH_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// A run of touching parts of one copy that came down in [`card_push`]'s
/// slots: the copy, the run's first byte in it, its parts (where each lies
/// in the slots' mapping, its length) and the send its plan went behind.
type PushedRun = (std::sync::Arc<crate::shadow::Shadow>, usize, Vec<(usize, usize)>, u64);

/// Merges a landing's [`PushedRun`]s into title memory and returns the
/// bytes written. With [`wide_merge`], copies are merged on several
/// threads: each copy's runs on one thread and in their order (a copy's
/// merges take its own lock one after another anyway), the copies spread
/// by their bytes. The 9e6137f run spent 54 ms a frame copying back on
/// the one thread that lands flights, merging run after run.
///
/// The caller gives the slots back only after this returns: every part is
/// read by then.
fn merge_pushed(runs: Vec<PushedRun>) -> usize {
    if runs.is_empty() {
        return 0;
    }
    let started = std::time::Instant::now();
    fn merge_all(runs: Vec<PushedRun>) -> usize {
        let mut title = Vec::new();
        let mut written = 0;
        for (shadow, lo, parts, sent_as) in runs {
            // SAFETY: inside the slots' mapping (checked when each part was
            // taken); the pass that filled the slots has run, and no later
            // one takes them before the landing says they are read, which
            // is after the caller is done.
            let parts: Vec<&[u8]> = parts.iter().map(|&(at, n)| unsafe { std::slice::from_raw_parts(at as *const u8, n) }).collect();
            let n = shadow.card_merge_relay(lo, &parts, sent_as, &mut title);
            if n == 0 {
                PUSH_STATS[5].fetch_add(1, Ordering::Relaxed);
            }
            PUSH_STATS[4].fetch_add(1, Ordering::Relaxed);
            written += n;
        }
        written
    }
    // The copies, each with its runs in their order and its bytes.
    let mut index: crate::fx::Map<usize, usize> = crate::fx::Map::default();
    let mut copies: Vec<(usize, Vec<PushedRun>)> = Vec::new();
    let wide = wide_merge();
    let written = if !wide {
        for run in &runs {
            index.entry(std::sync::Arc::as_ptr(&run.0) as usize).or_insert(0);
        }
        merge_all(runs)
    } else {
        for run in runs {
            let bytes: usize = run.2.iter().map(|p| p.1).sum();
            let i = *index.entry(std::sync::Arc::as_ptr(&run.0) as usize).or_insert_with(|| {
                copies.push((0, Vec::new()));
                copies.len() - 1
            });
            copies[i].0 += bytes;
            copies[i].1.push(run);
        }
        if copies.len() < 2 {
            merge_all(copies.into_iter().flat_map(|(_, runs)| runs).collect())
        } else {
            // Largest first, each to the thread with the fewest bytes so far.
            copies.sort_unstable_by_key(|c| std::cmp::Reverse(c.0));
            let mut bins: Vec<(usize, Vec<PushedRun>)> = (0..copies.len().min(8)).map(|_| (0, Vec::new())).collect();
            for (bytes, runs) in copies {
                let bin = bins.iter_mut().min_by_key(|b| b.0).expect("at least two");
                bin.0 += bytes;
                bin.1.extend(runs);
            }
            let total = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let jobs: Vec<Box<dyn FnOnce() + Send>> = bins
                .into_iter()
                .map(|(_, runs)| {
                    let total = total.clone();
                    Box::new(move || {
                        total.fetch_add(merge_all(runs), Ordering::Relaxed);
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
            total.load(Ordering::Relaxed)
        }
    };
    PUSH_STATS[6].fetch_add(index.len() as u64, Ordering::Relaxed);
    PUSH_STATS[7].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    written
}

/// The pass of [`card_push`]: one group of 64 threads per 4 KiB part of
/// the range's pieces; a part whose bit the compare set in `differs` takes
/// the next slot of the ring (none if the landing thread has not read the
/// slot a lap ago yet) and copies its bytes there, then writes the slot's
/// record. Offsets as `CARD_COMPARE` cuts them, from the dword at or below
/// the part's first byte.
pub(crate) const CARD_PUSH: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint tag;      // the plan's tag
    uint count_at; // the plan's count in `meta`, in bytes
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer slots : register(u1);
RWByteAddressBuffer meta : register(u2);
static const uint SLOTS = 32768u;
static const uint SLOT = 4112u;
static const uint RECORDS = 4160u;
groupshared uint taken;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint t : SV_GroupIndex) {
    uint index = g.x >> 4;
    uint part = g.x & 15u;
    // No return ahead of the barrier: the whole group reaches it.
    bool flagged = index < pieces && (differs.Load(out_at + index * 4u) & (1u << part)) != 0u;
    if (t == 0u) {
        taken = 0xFFFFFFFFu;
        if (flagged) {
            uint got;
            meta.InterlockedAdd(0, 1u, got);
            uint was;
            meta.InterlockedAdd(count_at, 1u, was);
            if (got - meta.Load(4) < SLOTS) taken = got;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint s = taken;
    if (s == 0xFFFFFFFFu) return;
    uint w = (first + index) * 65536u + part * 4096u;
    uint base = (w > skew ? w - skew : 0u) & ~3u;
    uint n = base < size ? min(SLOT, size - base) : 0u;
    uint at = (s & (SLOTS - 1u)) * SLOT;
    for (uint j = t * 4u; j < n; j += 256u) {
        slots.Store(at + j, now.Load(base + j));
    }
    if (t == 0u) {
        meta.Store2(RECORDS + (s & (SLOTS - 1u)) * 8u, uint2(tag, ((out_at >> 2) + index) << 4 | part));
    }
}
"#;

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}
unsafe impl Send for CardPush {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, sixteen threads per 64 KiB piece (cut at
/// multiples of 64 KiB of title memory, as `Shadow::card_piece_parts` cuts them),
/// each hashing one 4 KiB part of it (`crate::shadow::CARD_PARTS`):
/// `differs` gets a dword per piece, with bit `j` set where a byte of part
/// `j` differs (0: the piece is unchanged). Only the flagged parts are
/// fetched and merged: a shader that wrote a few words of a piece cost a
/// fetch and a merge of all 64 KiB of it (runs of the gameplay frame
/// fetched 1,100-1,900 pieces, 73-125 MB, a frame). The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a part flagged for a
/// byte just outside it costs a fetch that finds nothing.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
// The parts of the group's four pieces that differ.
groupshared uint masks[4];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi < 4u) masks[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint index = id.x >> 4;
    uint part = id.x & 15u;
    bool live = index < pieces;
    bool differ = false;
    if (live) {
        uint piece = first + index;
        uint lo = piece * 65536u + part * 4096u;
        lo = lo > skew ? lo - skew : 0u;
        uint end = piece * 65536u + (part + 1u) * 4096u;
        uint hi = end > skew ? min(end - skew, size) : 0u;
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
        [loop]
        for (uint at = lo; at < hi; at += 16u) {
            h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
        uint slot = (piece * 16u + part) * 16u;
        if (compare != 0u) {
            differ = any(h != hashes.Load4(slot));
            // With the copy-back on its own thread: the next compare of the
            // part finds what is written from here on.
            if (compare == 2u) hashes.Store4(slot, h);
        } else {
            hashes.Store4(slot, h);
        }
    }
    if (differ) {
        uint was;
        InterlockedOr(masks[gi >> 4], 1u << part, was);
    }
    GroupMemoryBarrierWithGroupSync();
    if (live && part == 0u && compare != 0u) differs.Store(out_at + index * 4u, masks[gi >> 4]);
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. Run 102 sent up 514
/// ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each. With [`card_push`], the
/// parts that came down in slots are merged a copy to a thread
/// ([`merge_pushed`]).
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`card_push`]: the plan's tag, and the number of its send of
    /// title bytes in the order of the queue (what its pushed parts were
    /// taken behind).
    push: Option<(u32, u64)>,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: crate::fx::Set<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: crate::fx::Set<usize>,
    /// Title-written textures whose staging buffer is copied into the host
    /// texture in this list's prelude if refreshed or not uploaded yet.
    texture_uploads: Vec<(
        std::sync::Arc<crate::resource::HostTexture>,
        std::sync::Arc<crate::texture::Staging>,
    )>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// `VICTORY_BVH_QUARANTINE` (or the builder's warm-up): the list records
    /// a stage of the structure builder, so nothing its shaders write is
    /// copied back into title memory. Copy-backs are declared per list, not
    /// per dispatch: another shader's declaration in the same list covered
    /// the builder's stores too, and the quarantine leaked
    /// (fxceless/victory-fork#935).
    quarantined: bool,
    /// Numbered title-memory copies (`crate::memory`) already refreshed /
    /// marked for shader write-back on this list since the last reset (or,
    /// for reads, since the last step that wrote title memory).
    memory_read_seen: usize,
    memory_write_seen: usize,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }

    /// Advances the list's title-memory copy watermarks to `total` (`writes`:
    /// whether this dispatch may write title memory) and returns
    /// `(read_from, write_from)`.
    fn memory_sync_range(&mut self, total: usize, writes: bool) -> (usize, usize) {
        let read_from = self.memory_read_seen.min(total);
        self.memory_read_seen = total;
        let write_from = if writes {
            let prev = self.memory_write_seen.min(total);
            self.memory_write_seen = total;
            prev
        } else {
            total
        };
        (read_from, write_from)
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and the parts the compare flagged there
/// (all bits where it did not run). The parts to fetch, under whichever
/// range: all of them flagged under any. Two ranges of a plan may hold
/// parts of the same 64 KiB piece; the compare hashes the whole piece for
/// each and stores the hashes, so the second of them may find them stored
/// already and not flag them.
fn flagged_parts(flags: &[(usize, u32, u32)]) -> crate::fx::Map<(usize, u32), u32> {
    let mut parts = crate::fx::Map::default();
    for &(copy, piece, mask) in flags.iter().filter(|f| f.2 != 0) {
        *parts.entry((copy, piece)).or_insert(0) |= mask;
    }
    parts
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_parts, join_copy_ranges, join_ranges, landable, touches_other};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A part flagged under one range of a plan is fetched under every
    /// range that holds its piece, with the parts flagged under the
    /// others; pieces of other copies and pieces flagged nowhere are not.
    #[test]
    fn a_part_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, parts 0 and 5 flagged under
        // the first, part 7 under the second; piece 4 not flagged; copy b:
        // piece 3 not flagged, piece 9 not compared (all parts).
        let flags = [(a, 3, 0b10_0001), (a, 4, 0), (a, 3, 0b1000_0000), (b, 3, 0), (b, 9, u32::MAX)];
        let fetch = flagged_parts(&flags);
        assert_eq!(fetch.get(&(a, 3)), Some(&0b1010_0001));
        assert!(!fetch.contains_key(&(a, 4)));
        assert!(!fetch.contains_key(&(b, 3)));
        assert_eq!(fetch.get(&(b, 9)), Some(&u32::MAX));
        assert_eq!(fetch.len(), 2);
    }

    /// Refreshing the same host copy that declared an in-flight shader write
    /// needs no CPU wait (its GPU buffer receives the write in queue order);
    /// refreshing an overlapping range of another host copy, or a range a
    /// read-back writes (`owner == 0`), must wait until that flight lands.
    #[test]
    fn refreshes_wait_only_for_writes_from_another_copy_or_readback() {
        let (copy_a, copy_b) = (0xaaaa_0000usize, 0xbbbb_0000usize);
        let mut written = Vec::new();
        let mut by_copy = Vec::new();
        join_ranges(
            &mut written,
            [(0x1000, 0x1800), (0x1800, 0x2000), (0x4000, 0x4100)],
        );
        join_copy_ranges(
            &mut by_copy,
            [
                (copy_a, 0x1000, 0x1800),
                (copy_a, 0x1600, 0x2000),
                (0, 0x4000, 0x4100),
            ],
        );
        assert_eq!(by_copy, vec![(0, 0x4000, 0x4100), (copy_a, 0x1000, 0x2000)]);
        // Same copy: never blocked by its own pending GPU writes.
        assert!(!touches_other(&written, &by_copy, copy_a, 0x0800, 0x3000));
        // Aliased copy over the same title memory: blocked until copy-back.
        assert!(touches_other(&written, &by_copy, copy_b, 0x1200, 0x1400));
        assert!(!touches_other(&written, &by_copy, copy_b, 0x2000, 0x3000));
        // Read-back (`owner == 0`): blocks every copy.
        assert!(touches_other(&written, &by_copy, copy_a, 0x4000, 0x4080));
        assert!(touches_other(&written, &by_copy, copy_b, 0x4080, 0x4200));
    }

    /// Two command lists interleaved with newly numbered title-memory copies:
    /// a list that ran earlier still refreshes and declares write-back for
    /// copies another list numbered meanwhile, and re-refreshes all copies
    /// after a title-memory copy or callback on the same list.
    #[test]
    fn memory_sync_range_covers_copies_numbered_by_other_lists() {
        let mut list_a = super::Deferred::default();
        let mut list_b = super::Deferred::default();

        // List A sees copies 0..10 (read + write).
        assert_eq!(list_a.memory_sync_range(10, true), (0, 0));
        // List B numbers 5 more copies (total 15) and only reads them.
        assert_eq!(list_b.memory_sync_range(15, false), (0, 15));
        // List A's next dispatch must still refresh & write-back copies 10..15.
        assert_eq!(list_a.memory_sync_range(15, true), (10, 10));
        // List B then dispatches a writer at total 17: reads 15..17, writes 0..17.
        assert_eq!(list_b.memory_sync_range(17, true), (15, 0));
        // A title-memory copy clears the read watermark (refresh again) without
        // re-declaring write-back for copies already covered.
        list_a.memory_read_seen = 0;
        assert_eq!(list_a.memory_sync_range(17, true), (0, 15));
        // Resetting the list clears both watermarks.
        list_a = super::Deferred::default();
        assert_eq!(list_a.memory_sync_range(17, true), (0, 0));
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// Title memory ranges written by its texture read-backs on landing.
    backs: Vec<(usize, usize)>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    backs: Vec<(usize, usize)>,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence signalled behind the flights before it (retaining a COM ref).
    Signal(crate::fence::FenceRef, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders and read-backs may write
    /// (address, end), in order and joined.
    written: Vec<(usize, usize)>,
    /// `written` by owner (`Shadow` pointer, or `0` for read-backs), joined
    /// per owner so an aliased `Shadow` refresh sees writes of other copies.
    by_copy: Vec<(usize, usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`relay`]: the host copies the flights being landed wrote
    /// (`by_copy` still names them until they have landed), for
    /// [`plan_forwards`].
    landing: Vec<std::sync::Arc<crate::shadow::Shadow>>,
}

impl Line {
    /// Title memory ranges of bottom-of-pipe writes still in flight or being
    /// landed, in order and joined.
    fn immed_writes(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let all = self
            .in_hand
            .iter()
            .map(|&(at, _, n)| (at, at.saturating_add(n)))
            .chain(self.line.iter().filter_map(|item| match *item {
                Late::Write(at, _, n) => Some((at, at.saturating_add(n))),
                _ => None,
            }));
        join_ranges(&mut out, all);
        out
    }
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    by_copy: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    landing: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture, another host copy).
static LATE_STATS: [std::sync::atomic::AtomicU64; 19] = [const { std::sync::atomic::AtomicU64::new(0) }; 19];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether title memory `[from, to)` lies in what flights in the air may
/// write (with [`late`]).
pub(crate) fn flights_write(from: usize, to: usize) -> bool {
    touches(&line().written, from, to)
}

/// Whether a constant buffer view of a table whose title memory flights may
/// write is filled on the card, behind them ([`HostList::card_constants`]),
/// instead of in upload memory at submission, which waits for them to land
/// (`VICTORY_FILL_ON_CARD=0` fills it as before).
pub(crate) fn fill_on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_ON_CARD").is_none_or(|v| v != "0"))
}

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// Whether `[from, to)` of `copy` overlaps an in-flight write belonging to
/// another host copy or to a read-back (`owner == 0`).
fn touches_other(
    written: &[(usize, usize)],
    by_copy: &[(usize, usize, usize)],
    copy: usize,
    from: usize,
    to: usize,
) -> bool {
    touches(written, from, to)
        && by_copy
            .iter()
            .any(|&(owner, a, b)| owner != copy && a < to && b > from)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// `(owner, from, to)` ranges joined per `owner` where they overlap or touch.
fn join_copy_ranges(
    ranges: &mut Vec<(usize, usize, usize)>,
    more: impl IntoIterator<Item = (usize, usize, usize)>,
) {
    ranges.extend(more.into_iter().filter(|r| r.1 < r.2));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize, usize)> = Vec::with_capacity(ranges.len());
    for &(owner, from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if last.0 == owner && from <= last.2 => last.2 = last.2.max(to),
            _ => out.push((owner, from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures, or
/// another host copy over the same bytes) and runs steps of ours that may
/// read or write it anywhere; what the flights write is not there yet.
fn late_blocked(runs: &[&HostList], forwarded: &[(usize, usize, usize)]) -> Option<usize> {
    let (written, by_copy, immed) = {
        let line = line();
        let immed = line.immed_writes();
        if line.written.is_empty() && immed.is_empty() {
            return None;
        }
        (line.written.clone(), line.by_copy.clone(), immed)
    };
    blocked_by(&written, &by_copy, &immed, runs, true, forwarded)
}

/// Whether a refresh of a host copy over bytes that another host copy's
/// shaders write in a flight still in the air takes those bytes from that
/// copy's buffer on the card, in queue order, instead of the submitting
/// thread waiting for the flight to land (`VICTORY_ALIAS_FORWARD=0`: it
/// waits, as before).
///
/// The wait (fxceless#191, "title memory read … another copy" on the line
/// of flights) was about 3 settles and most of 66 ms a frame in the
/// PalmVI run (main 5493624, RX 7900 XT), with the card about 40% busy
/// while the submitting thread stood. The bytes are already on the card:
/// the flight's submission comes first in the queue, so a copy recorded
/// ahead of this submission's lists finds what its shaders wrote, as
/// [`card_push`] passes what shaders wrote to card copies down in the
/// submission that wrote it instead of fetching it afterwards.
///
/// Exact only where title memory over the range is still what the writing
/// copy's snapshot took there (`Shadow::snapshot_is_title`): then the
/// writing copy's buffer holds what title memory will hold once the flight
/// is copied back. Anywhere else (the title wrote there since, a
/// read-back, bottom-of-pipe writes, two writers, the refreshing copy's
/// own shaders writing there too, in a flight or in this submission, a copy
/// in the card's memory or in place), it waits as before.
///
/// The copy runs in queue order; what the CPU puts into the refreshing
/// copy does not, it is there at once. So from the moment a range is
/// planned until the card has passed its copy ([`Handed`]) the CPU puts
/// nothing into that range of the refreshing copy (`Shadow::hand`): a page
/// the title wrote beside the range would else go over the bytes of an
/// earlier copy that lists still queued read, and bytes refreshed after
/// the flight landed would end up under a copy still queued, with the
/// page noted as taken. A later refresh of the range in that time takes
/// the bytes on the card again, or needs nothing where the writing copy
/// still holds what title memory holds, or waits for the card
/// ([`handed_blocks`]).
pub(crate) fn alias_forward() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_FORWARD").is_none_or(|v| v != "0"))
}

/// A range `[address, end)` of title memory that `from`'s shaders write in
/// a flight, copied on the card from `from`'s buffer into `to`'s ahead of
/// the submission's lists ([`alias_forward`]).
struct Forward {
    from: std::sync::Arc<crate::shadow::Shadow>,
    to: std::sync::Arc<crate::shadow::Shadow>,
    address: usize,
    end: usize,
}

/// Since the last frame-rate line, of [`alias_forward`]: refreshes that
/// took bytes on the card instead of waiting, ranges copied, their bytes,
/// refreshes that waited because the check refused them, submissions that
/// waited for the card to pass a copy still queued ([`handed_ready`]), and
/// (unused).
///
/// Then: ranges the CPU took back while their copy was still queued
/// ([`handed_ready`]), and of the ranges looked at for it (one in 16), the
/// bytes looked at and those of them the title held and the writing copy
/// did not (`Shadow::title_missing`).
static FORWARD_STATS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// A [`Forward`] the card may not have passed yet: `done` is the fence
/// value signalled behind the submission it was sent ahead of. Until then
/// the CPU leaves `[address, end)` of `to` alone (`Shadow::hand`), unless
/// the range is `open`: no flight writes it through `from` any more, and
/// each submission that refreshes it in `to` brings `from` up to date
/// after it ([`handed_ready`]), so that the copy still queued brings
/// nothing older than the CPU put there.
struct Handed {
    to: std::sync::Arc<crate::shadow::Shadow>,
    from: std::sync::Arc<crate::shadow::Shadow>,
    address: usize,
    end: usize,
    done: u64,
    open: bool,
}

/// The copies between host copies the card may not have passed
/// ([`Handed`]). Changed by the submitting thread, under the submission
/// lock.
static HANDED: Mutex<Vec<Handed>> = Mutex::new(Vec::new());

/// Gives each of `copies` the ranges `handed` still holds for it.
fn handed_set(handed: &[Handed], copies: &[std::sync::Arc<crate::shadow::Shadow>]) {
    use std::sync::Arc;
    for (i, copy) in copies.iter().enumerate() {
        if copies[..i].iter().any(|c| Arc::ptr_eq(c, copy)) {
            continue;
        }
        copy.hand(handed.iter().filter(|h| !h.open && Arc::ptr_eq(&h.to, copy)).map(|h| (h.address, h.end)).collect());
    }
}

/// Forgets the copies between host copies that `gone` names (those the
/// card has passed, or that were never sent): the CPU refreshes their
/// ranges again.
fn handed_drop(gone: impl Fn(&Handed) -> bool) {
    let mut handed = HANDED.lock().unwrap_or_else(|e| e.into_inner());
    if !handed.iter().any(&gone) {
        return;
    }
    let copies: Vec<_> = handed.iter().filter(|h| gone(h)).map(|h| h.to.clone()).collect();
    handed.retain(|h| !gone(h));
    handed_set(&handed, &copies);
}

/// Notes `forwards` as on their way, sent ahead of the submission whose
/// fence value is `done`: from here the CPU leaves their ranges of the
/// refreshing copies alone.
fn hand(forwards: &[Forward], done: u64) {
    use std::sync::Arc;
    if forwards.is_empty() {
        return;
    }
    let mut handed = HANDED.lock().unwrap_or_else(|e| e.into_inner());
    for f in forwards {
        // The bytes change on the card: a kept snapshot is not current.
        f.to.taking(f.address, f.end);
        match handed.iter_mut().find(|h| Arc::ptr_eq(&h.to, &f.to) && h.address == f.address && h.end == f.end) {
            Some(h) => (h.from, h.done, h.open) = (f.from.clone(), done, false),
            None => handed.push(Handed {
                to: f.to.clone(),
                from: f.from.clone(),
                address: f.address,
                end: f.end,
                done,
                open: false,
            }),
        }
    }
    let copies: Vec<_> = forwards.iter().map(|f| f.to.clone()).collect();
    handed_set(&handed, &copies);
}

/// `[from, to)` without the `covered` ranges.
fn uncovered(from: usize, to: usize, covered: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut covered: Vec<(usize, usize)> = covered.iter().copied().filter(|&(a, b)| a < to && b > from).collect();
    covered.sort_unstable();
    let (mut at, mut left) = (from, Vec::new());
    for (a, b) in covered {
        if a > at {
            left.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        left.push((at, to));
    }
    left
}

/// Whether a copy still queued on the card ([`Handed`]) fills part of
/// title range `[address, end)` of the copy at `copy`.
fn handed_into(copy: usize, address: usize, end: usize) -> bool {
    HANDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|h| std::sync::Arc::as_ptr(&h.to) as usize == copy && h.address < end && h.end > address)
}

/// Makes `runs` ready around the copies between host copies still queued
/// on the card ([`Handed`]). `Err` if the flights must be settled first;
/// else the ranges of writing copies to bring up to date once the
/// submission's refreshes are done.
///
/// Per queued copy whose range the submission refreshes (anywhere in the
/// span of the refreshing copy's refreshes: a flush brings up that much):
///
/// - where the submission takes the bytes on the card again (`forwards`),
///   nothing: the CPU leaves the range alone as before;
/// - elsewhere no flight writes the range any more (a refresh of it would
///   have waited, or be in `forwards`). The title's bytes go in on the
///   CPU as for any refresh (the range is opened), and the writing copy
///   is refreshed after it, so that it holds nothing older when the queued
///   copy runs. That needs the writing copy free there: not given to
///   shaders in a flight or in this submission, and not waiting for a
///   copy on the card itself. If it is not, and a refresh names the
///   range, the flights are settled; if only the span holds it, the CPU
///   leaves it alone;
/// - a range given to this submission's shaders needs its snapshot taken
///   from the bytes the queued copy brings: settled first.
fn handed_ready(
    runs: &[&HostList],
    forwards: &[Forward],
) -> Result<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>, ()> {
    use std::sync::Arc;
    let mut handed = HANDED.lock().unwrap_or_else(|e| e.into_inner());
    if handed.is_empty() {
        return Ok(Vec::new());
    }
    let ptr = |s: &Arc<crate::shadow::Shadow>| Arc::as_ptr(s) as usize;
    // What the queued copies write (copy, address, end) and read from.
    let into: Vec<(usize, usize, usize)> = handed.iter().map(|h| (ptr(&h.to), h.address, h.end)).collect();
    let froms: Vec<usize> = handed.iter().map(|h| ptr(&h.from)).collect();
    // Of the copies they concern: the refreshes, their span per copy, and
    // the ranges given to shaders.
    let mut steps: Vec<(usize, usize, usize)> = Vec::new();
    // Per copy, the disjoint ranges a flush brings up ([`Batch::add`]).
    let mut spans: std::collections::HashMap<usize, Vec<(usize, usize)>> = std::collections::HashMap::new();
    let mut given: Vec<(usize, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                Step::Refresh(shadow, address, len) => {
                    let (copy, from, to) = (ptr(shadow), *address, *address + *len);
                    if into.iter().any(|&(c, ..)| c == copy) {
                        steps.push((copy, from, to));
                        push_range(spans.entry(copy).or_default(), from, to);
                    }
                }
                Step::GpuWrite(shadow, address, len) => {
                    let copy = ptr(shadow);
                    if froms.contains(&copy) || into.iter().any(|&(c, ..)| c == copy) {
                        given.push((copy, *address, *address + *len));
                    }
                }
                _ => {}
            }
        }
    }
    let touches = |ranges: &[(usize, usize, usize)], copy: usize, a: usize, b: usize| {
        ranges.iter().any(|&(c, x, y)| c == copy && x < b && y > a)
    };
    if into.iter().any(|&(copy, a, b)| touches(&given, copy, a, b)) {
        return Err(());
    }
    let mut after = Vec::new();
    for h in handed.iter_mut() {
        let (to, from) = (ptr(&h.to), ptr(&h.from));
        let Some(ranges) = spans.get(&to) else {
            continue;
        };
        let again: Vec<(usize, usize)> =
            forwards.iter().filter(|f| Arc::ptr_eq(&f.to, &h.to)).map(|f| (f.address, f.end)).collect();
        let parts: Vec<(usize, usize)> = ranges
            .iter()
            .filter(|&&(first, last)| h.address.max(first) < h.end.min(last))
            .flat_map(|&(first, last)| uncovered(h.address.max(first), h.end.min(last), &again))
            .collect();
        if parts.is_empty() {
            continue;
        }
        let free =
            |&(a, b): &(usize, usize)| h.from.free(a, b) && !touches(&given, from, a, b) && !touches(&into, from, a, b);
        if parts.iter().all(free) {
            if !h.open {
                FORWARD_STATS[6].fetch_add(1, Ordering::Relaxed);
            }
            h.open = true;
            after.extend(parts.into_iter().map(|(a, b)| (h.from.clone(), a, b)));
        } else if parts.iter().any(|&(a, b)| !free(&(a, b)) && touches(&steps, to, a, b)) {
            return Err(());
        } else {
            h.open = false;
        }
    }
    let copies: Vec<_> = handed.iter().map(|h| h.to.clone()).collect();
    handed_set(&handed, &copies);
    Ok(after)
}

/// The most bytes of title memory a submission compares with snapshots to
/// forward ([`alias_forward`]); past it, the rest waits.
const FORWARD_CHECK_MOST: usize = 64 << 20;

/// Why [`plan_forwards`] left a refresh to wait, in the order it looks:
/// what each of [`FORWARD_REFUSED`] counts, for the line of flights.
const FORWARD_WHY: [&str; 17] = [
    "bottom-of-pipe writes there",
    "the refreshing copy is in the card's memory",
    "the refreshing copy is in place",
    "a read-back writes there",
    "two copies write the same bytes",
    "the refreshing copy's own shaders write there in a flight",
    "the refreshing copy's own shaders write there in this submission",
    "the writing copy is not among the flights' copies",
    "the writing copy is in the card's memory",
    "the writing copy is in place",
    "the range is not inside both copies",
    "the writing copy has no snapshot",
    "the journal cannot say what the title wrote beside the writing copy's shader writes",
    "title memory is not what the snapshot took (without VICTORY_LATE_WAIT=2 or 3 nothing puts the title's bytes in)",
    "the writing copy takes bytes on the card itself",
    "the writing copy is not write-watched",
    "the writing copy's flight landed meanwhile",
];

/// Since the last frame-rate line: the refreshes [`plan_forwards`] left to
/// wait, by why ([`FORWARD_WHY`]). The first run with the hand-off (main
/// 7447a7a, RX 7900 XT) had 6 to 16 of them a frame and not one refresh
/// that took its bytes on the card, and one count for all of them.
static FORWARD_REFUSED: [std::sync::atomic::AtomicU64; 17] = [const { std::sync::atomic::AtomicU64::new(0) }; 17];

/// Whether two of `pieces` (owner, from, to) belong to different copies
/// and hold the same bytes: which of them wrote last is not known, so the
/// refresh waits ([`plan_forwards`]).
fn tangled_writers(pieces: &[(usize, usize, usize)]) -> bool {
    pieces
        .iter()
        .enumerate()
        .any(|(i, &(o, a, b))| pieces[..i].iter().any(|&(p, x, y)| p != o && x < b && y > a))
}

/// With flights in the air: the refreshes of `runs` that another copy's
/// in-flight shader writes would block, and that can take those bytes on
/// the card instead ([`alias_forward`]): the copies to make, and the
/// refreshes (copy, address, end) that need not wait for them.
fn plan_forwards(runs: &[&HostList]) -> (Vec<Forward>, Vec<(usize, usize, usize)>) {
    use std::sync::Arc;
    let mut forwards = Vec::new();
    let mut freed = Vec::new();
    if !alias_forward() || !late() {
        return (forwards, freed);
    }
    let (written, by_copy, immed, owners) = {
        let line = line();
        if line.written.is_empty() {
            return (forwards, freed);
        }
        // The copies flights wrote: of those in the air, and of those
        // being landed (the card has run them; `by_copy` names them until
        // their bytes are in title memory).
        let mut owners: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        let flying = line.line.iter().filter_map(|item| match item {
            Late::Run(flight) => Some(flight.back.iter().map(|(shadow, _)| shadow)),
            _ => None,
        });
        for shadow in flying.flatten().chain(&line.landing) {
            if !owners.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                owners.push(shadow.clone());
            }
        }
        (line.written.clone(), line.by_copy.clone(), line.immed_writes(), owners)
    };
    // What this submission gives to shaders (copy, from, to): the snapshot
    // of such a range is taken on the CPU before the lists go, when a copy
    // on the card has not brought its bytes yet.
    let mut given: Vec<(usize, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                given.push((Arc::as_ptr(shadow) as usize, *address, *address + *len));
            }
        }
    }
    let mut checked = 0usize;
    for list in runs {
        for step in &list.deferred().copies {
            let Step::Refresh(to, address, len) = step else {
                continue;
            };
            let (from, end) = (*address, *address + *len);
            let copy = Arc::as_ptr(to) as usize;
            if freed.contains(&(copy, from, end)) || !touches_other(&written, &by_copy, copy, from, end) {
                continue;
            }
            let others: Vec<(usize, usize, usize)> =
                by_copy.iter().copied().filter(|&(o, a, b)| o != copy && a < end && b > from).collect();
            // The pieces of the refreshed range, each with the copy whose
            // shaders write it. Several copies may write one range, each
            // its own pieces: only where two of them write the same bytes
            // is it unknown which wrote last.
            let pieces: Vec<(usize, usize, usize)> =
                others.iter().map(|&(o, a, b)| (o, a.max(from), b.min(end))).collect();
            let over = |ranges: &[(usize, usize, usize)]| {
                ranges.iter().any(|&(o, a, b)| o == copy && others.iter().any(|&(_, x, y)| x < b && y > a))
            };
            let tangled = tangled_writers(&pieces);
            // Why it waits, if it does ([`FORWARD_WHY`]), and the writing
            // copy of each piece.
            let mut writers: Vec<Arc<crate::shadow::Shadow>> = Vec::with_capacity(pieces.len());
            let mut why = if touches(&immed, from, end) {
                Some(0)
            } else if to.carded() {
                Some(1)
            } else if !to.plain() {
                Some(2)
            } else if pieces.iter().any(|&(o, ..)| o == 0) {
                Some(3)
            } else if tangled {
                Some(4)
            } else if over(&by_copy) {
                Some(5)
            } else if over(&given) {
                Some(6)
            } else {
                None
            };
            if why.is_none() {
                for &(o, a, b) in &pieces {
                    let Some(w) = owners.iter().find(|s| Arc::as_ptr(s) as usize == o) else {
                        why = Some(7);
                        break;
                    };
                    let (ws, wl) = w.span();
                    let (ts, tl) = to.span();
                    if w.carded() {
                        why = Some(8);
                    } else if !w.plain() {
                        why = Some(9);
                    } else if (a as u64) < ws.max(ts) || (b as u64) > (ws + wl).min(ts + tl) {
                        why = Some(10);
                    } else if !w.snapshot_taken() {
                        why = Some(11);
                    } else if handed_into(o, a, b) {
                        // Its own bytes are coming from the card: what it
                        // holds now is neither the title's nor the writer's.
                        why = Some(14);
                    } else if !late_exact() {
                        // Nothing puts the title's bytes in beside shader
                        // writes: only a range the title has not written
                        // since the snapshot can go.
                        if checked + (b - a) > FORWARD_CHECK_MOST || !w.snapshot_is_title(a, b) {
                            why = Some(13);
                        }
                        checked += b - a;
                    } else if w.unwatched() {
                        why = Some(15);
                    } else if !w.pending_holds(a, b) {
                        why = Some(16);
                    } else {
                        // What the title wrote there since the writing copy
                        // last took it goes into that copy and its snapshot
                        // beside the shader writes, as for a refresh of that
                        // copy under flights ([`stale_under_flights`]). The
                        // copy then holds what title memory will hold after
                        // the copy-back, as far as the title has written by
                        // now.
                        let mut runs_left = 512;
                        if w.merge_under_writes(a, b - a, &mut runs_left) {
                            why = Some(12);
                        } else if !w.pending_holds(a, b) {
                            // The copy-back's thread ended a range between
                            // the question above and the merge, which then
                            // put in nothing there (it answers false for a
                            // range no shaders write). Ranges only end
                            // meanwhile, so one that is held now was held
                            // for the merge.
                            why = Some(16);
                        } else {
                            // One range in 16 is looked at for the line of
                            // flights: the title's bytes the copy lacks.
                            static NTH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                            if NTH.fetch_add(1, Ordering::Relaxed).is_multiple_of(16)
                                && let Some(missing) = w.title_missing(a, b.min(a + (1 << 20)))
                            {
                                FORWARD_STATS[7].fetch_add((b.min(a + (1 << 20)) - a) as u64, Ordering::Relaxed);
                                FORWARD_STATS[8].fetch_add(missing as u64, Ordering::Relaxed);
                            }
                        }
                    }
                    if why.is_some() {
                        break;
                    }
                    writers.push(w.clone());
                }
            }
            if let Some(why) = why {
                FORWARD_STATS[3].fetch_add(1, Ordering::Relaxed);
                FORWARD_REFUSED[why].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            FORWARD_STATS[0].fetch_add(1, Ordering::Relaxed);
            for (&(_, a, b), writer) in pieces.iter().zip(writers) {
                FORWARD_STATS[1].fetch_add(1, Ordering::Relaxed);
                FORWARD_STATS[2].fetch_add((b - a) as u64, Ordering::Relaxed);
                forwards.push(Forward {
                    from: writer,
                    to: to.clone(),
                    address: a,
                    end: b,
                });
            }
            freed.push((copy, from, end));
        }
    }
    (forwards, freed)
}

/// [`late_blocked`] against `written` (joined), `by_copy` (per owner), and
/// `immed` (bottom-of-pipe writes); `count` whether the kind of read is
/// counted for the frame line; `forwarded` the refreshes (copy, address,
/// end) that take the in-flight bytes on the card ([`plan_forwards`]).
fn blocked_by(
    written: &[(usize, usize)],
    by_copy: &[(usize, usize, usize)],
    immed: &[(usize, usize)],
    runs: &[&HostList],
    count: bool,
    forwarded: &[(usize, usize, usize)],
) -> Option<usize> {
    if written.is_empty() && immed.is_empty() {
        return None;
    }
    let any_written = |from: usize, to: usize| touches(written, from, to) || touches(immed, from, to);
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if any_written(src, src + len) || any_written(dst, dst + len) {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len) => {
                    if any_written(src, src + len) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            static NAMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                            if NAMED.fetch_add(1, Ordering::Relaxed) < 12 {
                                let copy = crate::shadow::find(src as u64, len as u64).map(|s| s.span());
                                tracing::info!(
                                    src = format_args!("{src:#x}"),
                                    len,
                                    copy = ?copy.map(|(a, l)| format!("{a:#x}+{l:#x}")),
                                    "a fill of upload memory reads what a flight writes"
                                );
                            }
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if any_written(at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Refresh(shadow, address, len) => {
                    let (from, to) = (*address, *address + *len);
                    let copy = std::sync::Arc::as_ptr(shadow) as usize;
                    if forwarded.contains(&(copy, from, to)) {
                        continue;
                    }
                    if touches_other(written, by_copy, copy, from, to) || touches(immed, from, to) {
                        if count {
                            LATE_STATS[18].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    None
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> bool {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                    Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                    None => spans.push((shadow.clone(), *address, *address + *len)),
                }
            }
        }
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there.
        let mut runs_left = 512;
        return spans.iter().any(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
    }
    spans.iter().any(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from))
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Returns `Ok(())`
/// if queued, or `Err(fence)` if the caller should set the fence now.
pub(crate) fn late_signal(
    fence: crate::fence::FenceRef,
    value: u64,
) -> Result<(), crate::fence::FenceRef> {
    if !late() {
        return Err(fence);
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return Err(fence);
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    Ok(())
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
/// The decision of [`late_met`] (fxceless #785): `Some(true)` if the last
/// queued write of the whole value meets `holds` and nothing else may write
/// there; `Some(false)` if it surely is not met; `None` if a shader, a
/// timestamp or a write of another width overlaps, so the flights are
/// settled and memory is asked again.
fn eval_late_met(
    written: &[(usize, usize)],
    writes: impl IntoIterator<Item = (usize, crate::queue::Written, usize)>,
    address: usize,
    bytes: usize,
    holds: impl Fn(u64) -> bool,
) -> Option<bool> {
    let end = address.saturating_add(bytes);
    let mut unsure = touches(written, address, end);
    let mut last_exact: Option<bool> = None;
    for (at, what, n) in writes {
        if at >= end || address >= at.saturating_add(n) {
            continue;
        }
        match what {
            crate::queue::Written::Value(v) if at == address && n >= bytes => {
                last_exact = Some(holds(v));
            }
            _ => unsure = true,
        }
    }
    if unsure {
        None
    } else {
        Some(last_exact.unwrap_or(false))
    }
}

pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let decided = {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() && !line.settling {
            return false;
        }
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().copied().chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((*at, *what, *n)),
            _ => None,
        }));
        eval_late_met(&line.written, writes, address, bytes, |v| wait.holds_for(v))
    };
    match decided {
        Some(true) => {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            true
        }
        Some(false) => false,
        None => {
            settle_now(WHY_VALUE);
            wait.met()
        }
    }
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed = std::mem::take(&mut line().landed);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                line.landing = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Run(flight) => Some(flight.back.iter().map(|(shadow, _)| shadow.clone())),
                        _ => None,
                    })
                    .flatten()
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards).
fn relay_hand_back(back: &[Declared]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    if work.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let scanning = std::time::Instant::now();
    // Looked through without a lock: a piece that changes meanwhile (a
    // later flight writing, the title's bytes going in) is found now or by
    // that flight's landing, and the merge looks again under the lock.
    let found = crate::shadow::find_all(&work);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, from, to), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::compared(*from, to - from, written);
        let _ = (from, to);
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}, another copy {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[18],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    if alias_forward() {
        let f: Vec<f64> = FORWARD_STATS.iter().map(|f| f.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "refreshes over another copy's in-flight shader writes, per frame: {:.1} took the bytes on the card ({:.1} ranges, {:.1} MB) \
             instead of waiting, {:.1} waited (title wrote there since, no snapshot, a read-back, two writers, the copy's own \
             shaders write there, or a copy not plain), {:.1} submissions waited for the card to pass such a copy still queued, \
             {:.1} ranges went back to the CPU while their copy was queued; of {:.2} MB looked at (one range in 16), {:.0} bytes \
             were the title's and not in the writing copy (VICTORY_ALIAS_FORWARD=0: all wait)",
            f[0],
            f[1],
            f[2] / 1048576.0,
            f[3],
            f[4],
            f[6],
            f[7] / 1048576.0,
            f[8],
        );
        let why: Vec<String> = FORWARD_REFUSED
            .iter()
            .zip(FORWARD_WHY)
            .map(|(n, why)| (n.swap(0, Ordering::Relaxed) as f64 / frames, why))
            .filter(|&(n, _)| n > 0.0)
            .map(|(n, why)| format!("{n:.1} {why}"))
            .collect();
        if !why.is_empty() {
            tracing::info!("of the refreshes that waited for another copy's shader writes, per frame, by why: {}", why.join(", "));
        }
    }
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame (VICTORY_RELAY=1): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it)",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if crate::shadow::copy_compare() {
            let [all, same] = [0, 1].map(|i| crate::shadow::COPY_COMPARED[i].swap(0, Ordering::Relaxed) as f64 / 1e6 / frames);
            tracing::info!(
                "refreshes compared per frame (VICTORY_COPY_COMPARE=1): {all:.1} MB the title wrote, {same:.1} MB of it \
                 the same bytes as before (not put in, not sent up)"
            );
        }
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
    }
}

#[cfg(test)]
mod upload_pool_tests {
    use super::{give_upload, host, upload_room};

    /// The room of an upload buffer holds what was asked for, wastes an
    /// eighth at most above 1 MiB (64 KiB below), and is its own room.
    #[test]
    fn an_upload_buffers_room_is_a_little_more_than_asked() {
        for size in [1u64, 4352, 65_536, 65_537, 1 << 20, (1 << 20) + 1, 5_000_000, 174 << 20] {
            let room = upload_room(size);
            assert!(room >= size, "{size}");
            assert!(room - size < (64 << 10).max(size / 4), "{size} in {room}");
            assert_eq!(upload_room(room), room, "{size}");
        }
        assert_eq!(upload_room(0), 64 << 10);
    }

    /// A buffer given back is the one the next ask of its room takes,
    /// zeroed; an ask of another room makes its own.
    #[test]
    fn a_given_back_upload_buffer_is_taken_again_zeroed() {
        use windows::core::Interface;
        let Some(host) = host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        // A room no other test asks for.
        let size = (3 << 20) + 12_345;
        let (buffer, data, room) = host.upload_pooled(size).expect("a buffer");
        assert_eq!(room, upload_room(size));
        // SAFETY: the mapping holds `room` bytes.
        unsafe { std::ptr::write_bytes(data, 0xAB, room as usize) };
        let raw = buffer.as_raw() as usize;
        give_upload(buffer, data, room);
        let (other, _, other_room) = host.upload_pooled(size / 2).expect("a buffer");
        assert_ne!(other.as_raw() as usize, raw);
        assert_ne!(other_room, room);
        let (again, data, again_room) = host.upload_pooled(size - 100).expect("a buffer");
        assert_eq!(again.as_raw() as usize, raw);
        assert_eq!(again_room, room);
        // SAFETY: as above.
        let bytes = unsafe { std::slice::from_raw_parts(data, room as usize) };
        assert!(bytes.iter().all(|&b| b == 0));
    }
}

#[cfg(test)]
mod forward_tests {
    use super::{Forward, host};
    use std::sync::Arc;

    /// The card copies a range of one copy's buffer into another's
    /// ([`super::alias_forward`]), and nothing outside it; a snapshot is
    /// title memory only where it holds the title's bytes.
    #[test]
    fn a_forward_takes_another_copys_bytes_on_the_card() {
        let Some(host) = host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        const LEN: usize = 3 * 4096;
        // The title memory both copies are of.
        let title: Vec<u8> = (0..LEN).map(|i| (i % 7) as u8).collect();
        let start = title.as_ptr() as usize;
        let (a_buffer, a_data) = host.shared_buffer(LEN as u64).expect("a buffer");
        let (b_buffer, b_data) = host.shared_buffer(LEN as u64).expect("a buffer");
        // The writer holds the title's bytes, but for one of its last page,
        // when shaders are given all of it: that is its snapshot.
        // SAFETY: the mappings hold LEN bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(title.as_ptr(), a_data, LEN);
            *a_data.add(2 * 4096) ^= 1;
            std::ptr::write_bytes(b_data, 0xEE, LEN);
        }
        let a = Arc::new(crate::shadow::Shadow::for_test(a_buffer, a_data, start as u64, LEN as u64, None));
        let b = Arc::new(crate::shadow::Shadow::for_test(b_buffer, b_data, start as u64, LEN as u64, None));
        assert!(!a.snapshot_is_title(start + 4096, start + 2 * 4096));
        a.writing(start, LEN);
        crate::shadow::take_snapshots();
        // What the shaders write.
        // SAFETY: as above.
        unsafe {
            for i in 0..LEN {
                *a_data.add(i) = (i % 251) as u8;
            }
        }
        assert!(a.snapshot_is_title(start + 4096, start + 2 * 4096));
        assert!(!a.snapshot_is_title(start + 4096, start + 3 * 4096));
        assert!(!b.snapshot_is_title(start, start + 4096));
        let forwards = [Forward {
            from: a.clone(),
            to: b.clone(),
            address: start + 4096,
            end: start + 2 * 4096,
        }];
        assert!(host.send_forwards(&forwards, u64::MAX));
        // Wait for the card behind the copy.
        // SAFETY: plain creation, signal and query.
        unsafe {
            let fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence =
                host.device.CreateFence(0, windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE).expect("a fence");
            host.queue.Signal(&fence, 1).expect("signalled");
            let waited = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(waited.elapsed() < std::time::Duration::from_secs(10), "the card did not finish");
                std::thread::yield_now();
            }
        }
        let got = b.bytes_for_test();
        for (i, &byte) in got.iter().enumerate() {
            let want = if (4096..2 * 4096).contains(&i) { (i % 251) as u8 } else { 0xEE };
            assert_eq!(byte, want, "byte {i}");
        }
        drop(title);
    }

    /// While a copy between host copies is queued on the card
    /// (`Shadow::hand`), a refresh puts nothing into its range of the
    /// refreshing copy, to the byte, and does not note its pages as taken:
    /// once the card has passed it, the title's bytes go in. And the
    /// writing copy holds title memory only where its bytes are the
    /// title's.
    #[test]
    fn a_refresh_leaves_a_range_alone_while_its_copy_is_queued() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGE: usize = crate::watch::PAGE;
        const LEN: usize = 3 * PAGE;
        let Some(host) = host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }.cast::<u8>();
        assert!(!title.is_null());
        let start = title as usize;
        let (a_buffer, a_data) = host.shared_buffer(LEN as u64).expect("a buffer");
        let (b_buffer, b_data) = host.shared_buffer(LEN as u64).expect("a buffer");
        // The queued copy's range: inside the middle page, not all of it.
        let (from, to) = (PAGE + 100, 2 * PAGE - 100);
        // SAFETY: the allocation and the mappings hold LEN bytes.
        unsafe {
            for i in 0..LEN {
                *title.add(i) = (i % 7) as u8 + 1;
                *a_data.add(i) = (i % 7) as u8 + 1;
            }
            // The writing copy differs from title memory in one byte of
            // the range.
            *a_data.add(from + 50) ^= 0x80;
            std::ptr::write_bytes(b_data, 0xEE, LEN);
        }
        let a = crate::shadow::Shadow::for_test(a_buffer, a_data, start as u64, LEN as u64, None);
        let b = crate::shadow::Shadow::for_test(b_buffer, b_data, start as u64, LEN as u64, None);
        b.hand(vec![(start + from, start + to)]);
        b.update(start, LEN);
        for (i, &byte) in b.bytes_for_test().iter().enumerate() {
            let want = if (from..to).contains(&i) { 0xEE } else { (i % 7) as u8 + 1 };
            assert_eq!(byte, want, "byte {i} with the copy queued");
        }
        assert!(!a.holds_title(start + from, start + to));
        assert!(a.holds_title(start + from + 51, start + to));
        // SAFETY: inside the mapping.
        unsafe { *a_data.add(from + 50) ^= 0x80 };
        assert!(a.holds_title(start + from, start + to));
        assert!(!a.holds_title(start + from, start + LEN + 1));
        // The card has passed the copy.
        b.hand(Vec::new());
        b.update(start, LEN);
        for (i, &byte) in b.bytes_for_test().iter().enumerate() {
            assert_eq!(byte, (i % 7) as u8 + 1, "byte {i} with the copy passed");
        }
        drop((a, b));
        // SAFETY: the allocation above; nothing names it any more.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
    }

    /// What the hand-off asks of a writing copy: whether declared shader
    /// writes hold a range whole (`pending_holds`), whether none touch it
    /// (`free`), and how many of the title's bytes the copy lacks
    /// (`title_missing`: the title's and not the snapshot's, where the
    /// card has not written).
    #[test]
    fn a_writing_copy_says_what_its_declared_ranges_hold() {
        const LEN: usize = 3 * 4096;
        let Some(host) = host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        let mut title: Vec<u8> = (0..LEN).map(|i| (i % 7) as u8 + 1).collect();
        let start = title.as_ptr() as usize;
        let (buffer, data) = host.shared_buffer(LEN as u64).expect("a buffer");
        // SAFETY: the mapping holds LEN bytes.
        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr(), data, LEN) };
        let a = crate::shadow::Shadow::for_test(buffer, data, start as u64, LEN as u64, None);
        assert!(a.free(start, start + LEN));
        assert!(!a.pending_holds(start + 4096, start + 2 * 4096));
        // Shaders are given the middle page; its snapshot is the copy's
        // bytes, which are the title's.
        a.writing(start + 4096, 4096);
        crate::shadow::take_snapshots();
        assert!(a.pending_holds(start + 4096, start + 2 * 4096));
        assert!(a.pending_holds(start + 4096 + 10, start + 4096 + 20));
        assert!(!a.pending_holds(start + 4095, start + 4097));
        assert!(!a.pending_holds(start, start + LEN));
        assert!(a.free(start, start + 4096));
        assert!(a.free(start + 2 * 4096, start + LEN));
        assert!(!a.free(start + 4095, start + 4097));
        assert_eq!(a.title_missing(start + 4096, start + 2 * 4096), Some(0));
        // The title writes two bytes there; the card has written one of
        // them too.
        title[4096 + 5] ^= 0x40;
        title[4096 + 9] ^= 0x40;
        // SAFETY: inside the mapping.
        unsafe { *data.add(4096 + 9) ^= 0x01 };
        assert_eq!(a.title_missing(start + 4096, start + 2 * 4096), Some(1));
        assert_eq!(a.title_missing(start + 4096, start + LEN + 1), None);
        drop(a);
        drop(title);
    }

    /// Pieces of one refreshed range written by different copies are
    /// forwarded where each copy has its own bytes, and wait where two of
    /// them hold the same byte ([`super::tangled_writers`]).
    #[test]
    fn two_writers_tangle_only_where_they_write_the_same_bytes() {
        use super::tangled_writers;
        let (a, b) = (0xaaaa_0000usize, 0xbbbb_0000usize);
        assert!(!tangled_writers(&[]));
        assert!(!tangled_writers(&[(a, 10, 20)]));
        // Side by side, and the same copy twice.
        assert!(!tangled_writers(&[(a, 10, 20), (b, 20, 30)]));
        assert!(!tangled_writers(&[(a, 10, 20), (a, 15, 30)]));
        // The same byte from two copies.
        assert!(tangled_writers(&[(a, 10, 20), (b, 19, 30)]));
        assert!(tangled_writers(&[(a, 10, 20), (a, 40, 50), (b, 0, 11)]));
    }

    #[test]
    fn uncovered_is_the_range_without_what_covers_it() {
        use super::uncovered;
        assert_eq!(uncovered(10, 20, &[]), [(10, 20)]);
        assert_eq!(uncovered(10, 20, &[(0, 30)]), []);
        assert_eq!(uncovered(10, 20, &[(12, 14), (0, 11), (18, 40)]), [(11, 12), (14, 18)]);
        assert_eq!(uncovered(10, 20, &[(20, 30), (0, 10)]), [(10, 20)]);
        assert_eq!(uncovered(10, 20, &[(13, 15), (14, 16)]), [(10, 13), (16, 20)]);
    }
}

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (on by default;
/// `VICTORY_JOIN_COPIES=0` turns it off).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
/// Buffer-to-texture copies whose source this list's shaders write, staged
/// on the GPU ([`HostList::staged_written_source`]), since the start.
pub(crate) static STAGED_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let range = (*address, *address + *len);
                match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                    Some((_, ranges)) => {
                        if !ranges.contains(&range) {
                            ranges.push(range);
                        }
                    }
                    None => back.push((shadow.clone(), vec![range])),
                }
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        if *rows == 0 || *row == 0 {
            continue;
        }
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run. Title memory is written through `write_mapped` after the
        // watch is told, so a page unmapped or guarded while the list flew
        // is skipped instead of faulting (PalmVIDev 2ef001c8, from
        // fxceless/victory-fork#565).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                let src_pitch = row.next_multiple_of(256);
                let span = (rows - 1).saturating_mul(*pitch).saturating_add(*row);
                let write = TitleWrite::open(*at, span);
                for line in 0..*rows {
                    let Some(dst) = line.checked_mul(*pitch).and_then(|off| at.checked_add(off)) else {
                        break;
                    };
                    write.put(dst, from.add(line * src_pitch), *row);
                }
                drop(write);
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host's pixel shaders read `SV_Barycentrics` and each
    /// vertex's attribute values (`GetAttributeAtVertex`), which a pixel
    /// shader linked to a mesh shader whose outputs are packed needs
    /// (`victory_shader::emit::Link`); logged once.
    pub(crate) fn barycentrics(&self) -> bool {
        static BARYCENTRICS: OnceLock<bool> = OnceLock::new();
        *BARYCENTRICS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let barycentrics = ok && options.BarycentricsSupported.as_bool();
            tracing::info!(barycentrics, "host barycentrics support");
            barycentrics
        })
    }

    /// Whether the host GPU lets a view of a typed texture use another
    /// format of the same family (D3D12's relaxed casting rules, reported
    /// in Options3); logged once.
    pub(crate) fn typed_format_casting(&self) -> bool {
        static CASTING: OnceLock<bool> = OnceLock::new();
        *CASTING.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.CastingFullyTypedFormatSupported.as_bool();
            tracing::info!(supported, "host typed format casting support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        crate::census::made(8, 0);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// Forgets the copies between host copies the card has passed
    /// ([`Handed`]).
    fn handed_passed(&self) {
        // SAFETY: a plain query.
        let passed = unsafe { self.fence.GetCompletedValue() };
        handed_drop(|h| h.done <= passed);
    }

    /// Records and sends the copies of [`alias_forward`]: each range from
    /// the writing copy's buffer into the refreshing copy's, on the card, in
    /// a list of its own sent now (behind the flights whose shaders wrote
    /// the bytes, ahead of the submission's lists). `done` is the fence
    /// value signalled behind the submission: the list and both copies are
    /// kept until the card has passed it. False if no list could be had.
    fn send_forwards(&self, forwards: &[Forward], done: u64) -> bool {
        type Kept = (u64, Prelude, Vec<std::sync::Arc<crate::shadow::Shadow>>);
        struct Pool(Vec<Kept>);
        // SAFETY: the lists and allocators are used by one submitting
        // thread at a time (under the submission lock).
        unsafe impl Send for Pool {}
        static POOL: Mutex<Pool> = Mutex::new(Pool(Vec::new()));
        // SAFETY: a plain query.
        let passed = unsafe { self.fence.GetCompletedValue() };
        let reused = {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            // What the card has passed holds no copy any longer.
            for kept in pool.0.iter_mut().filter(|k| k.0 <= passed) {
                kept.2.clear();
            }
            let at = pool.0.iter().position(|k| k.0 <= passed);
            at.map(|at| pool.0.swap_remove(at).1)
        };
        let Some(prelude) = reused.or_else(|| self.make_prelude()) else {
            return false;
        };
        let (allocator, list) = &prelude;
        // SAFETY: a list the card is done with; buffers of live copies
        // (kept below until `done`), in the common state (promoted to the
        // copy states for the list and back at its end).
        let sent = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                for f in forwards {
                    let (from_start, _) = f.from.span();
                    let (to_start, _) = f.to.span();
                    list.CopyBufferRegion(
                        &f.to.buffer,
                        f.address as u64 - to_start,
                        &f.from.buffer,
                        f.address as u64 - from_start,
                        (f.end - f.address) as u64,
                    );
                }
                list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                Ok(())
            })()
        };
        if let Err(e) = sent {
            tracing::warn!(error = %e, "copies between host copies on the card not sent");
            return false;
        }
        let held = forwards.iter().flat_map(|f| [f.from.clone(), f.to.clone()]).collect();
        POOL.lock().unwrap_or_else(|e| e.into_inner()).0.push((done, prelude, held));
        true
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                buffer_states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                hoisted: Mutex::default(),
                hoist_mark: std::sync::atomic::AtomicU64::new(0),
                card_constants: Mutex::default(),
                _alive: Counted::new(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                CARD_COMPARE.as_ptr().cast(),
                CARD_COMPARE.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                push: if relay() && card_push() { self.make_card_push() } else { None },
            })
        }
    }

    /// [`CardPush`]: `None` (the parts are fetched as before) if the pass
    /// does not compile or the memory is not there.
    fn make_card_push(&self) -> Option<CardPush> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointers are ours.
        let compiled = unsafe {
            D3DCompile(
                CARD_PUSH.as_ptr().cast(),
                CARD_PUSH.len(),
                windows::core::s!("card push"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        let Some(code) = code.filter(|_| compiled.is_ok()) else {
            let said = errors
                // SAFETY: the blob's own bytes.
                .map(|e| unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned() })
                .unwrap_or_default();
            tracing::warn!(error = %said, "the push of shader writes from the card did not compile: they are fetched as before");
            return None;
        };
        let slots = self.shared_buffer(u64::from(PUSH_SLOTS) * u64::from(PUSH_SLOT))?;
        name(&slots.0, || "parts shaders wrote, pushed from the card".into());
        let meta = self.shared_buffer(u64::from(PUSH_RECORDS) + 8 * u64::from(PUSH_SLOTS))?;
        name(&meta.0, || "parts shaders wrote, pushed from the card: counts and records".into());
        // SAFETY: the mapping holds the counts and records; nothing on the
        // card uses it yet. Records start with tag 0, which no plan has.
        unsafe { std::ptr::write_bytes(meta.1, 0, (PUSH_RECORDS + 8 * PUSH_SLOTS) as usize) };
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes).ok()?;
            let pso: ID3D12PipelineState = self
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            Some(CardPush {
                root,
                pso,
                slots: (slots.0, slots.1 as usize),
                meta: (meta.0, meta.1 as usize),
                next_tag: 1,
                cursor: 0,
            })
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_parts`] and
    /// `relay`).
    fn card_before(&self, runs: &[&HostList], flown: bool) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let mut declared: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && shadow.carded()
                {
                    let range = (*address, *address + *len);
                    match declared.iter_mut().find(|(s, _)| Arc::ptr_eq(s, shadow)) {
                        Some((_, ranges)) => {
                            if !ranges.contains(&range) {
                                ranges.push(range);
                            }
                        }
                        None => declared.push((shadow.clone(), vec![range])),
                    }
                }
            }
        }
        // Experiment (`VICTORY_SKIP_WC_COMPARE=1`, `crate::census`): copies
        // whose declared writes all lie in write-combining title memory are
        // left out of the hashing and the compare; nothing of them is
        // fetched back.
        if crate::census::skip_wc_compare() {
            let before = declared.len();
            declared.retain(|(_, ranges)| !crate::census::all_wc(ranges));
            crate::census::UNCOMPARED.fetch_add((before - declared.len()) as u64, Ordering::Relaxed);
        }
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        if declared.is_empty() && dirty.is_empty() {
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Default::default()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for (shadow, updates) in &dirty {
                        let ranges = &updates.upload;
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        crate::census::sent(
                            shadow.title_start(),
                            ranges.iter().map(|&(a, b)| (b - a) as u64).sum(),
                            ranges.len() as u64,
                        );
                        let Some(before) = before else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. Hash pieces just uploaded or already
                        // merged from the card, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        if !ranges.is_empty() {
                            list.ResourceBarrier(&[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            updates.hash_ranges().iter().map(|&(a, b)| shadow.card_run(a, b)).collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            let constants = [run.first, run.skew, size as u32, run.pieces, 0, 0];
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.Dispatch((run.pieces * crate::shadow::CARD_PARTS).div_ceil(64), 1, 1);
                        }
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            push: None,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // Sent ahead with the relay: the flagged parts go down in this
            // list too ([`card_push`]), under a tag of the plan's own.
            let tag = match work.push.as_mut() {
                Some(push) if flown => {
                    let tag = push.next_tag;
                    push.next_tag = push.next_tag.wrapping_add(1).max(1);
                    // SAFETY: inside the meta mapping; the plan that had
                    // this count last (PUSH_TAGS plans ago) has landed.
                    unsafe {
                        ((push.meta.1 + (PUSH_COUNTS + 4 * (tag % PUSH_TAGS)) as usize) as *mut u32).write_volatile(0);
                    }
                    plan.push = Some((tag, sent_as));
                    Some(tag)
                }
                _ => None,
            };
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        let constants = [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, if flown { 2 } else { 1 }];
                        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.Dispatch((r.span.pieces * crate::shadow::CARD_PARTS).div_ceil(64), 1, 1);
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    if let (Some(tag), Some(push)) = (tag, work.push.as_ref()) {
                        // The compare's results, first.
                        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&push.root);
                        list.SetPipelineState(&push.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, push.slots.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, push.meta.0.GetGPUVirtualAddress());
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                continue;
                            };
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            // A group per part, at most 65,535 groups a dispatch.
                            const MOST: u32 = 65_535 / crate::shadow::CARD_PARTS;
                            let mut done = 0;
                            while done < r.span.pieces {
                                let pieces = (r.span.pieces - done).min(MOST);
                                let constants = [
                                    r.span.first + done,
                                    r.span.skew,
                                    size as u32,
                                    pieces,
                                    (r.at + done) * 4,
                                    tag,
                                    PUSH_COUNTS + 4 * (tag % PUSH_TAGS),
                                ];
                                list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                                list.Dispatch(pieces * crate::shadow::CARD_PARTS, 1, 1);
                                done += pieces;
                            }
                        }
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_parts`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up.
    fn relay_card_back(&self, plans: &[&CardPlan], fetch: &mut Option<RelayFetch>) {
        // The results, and with [`card_push`] the slots each plan's parts
        // went down to (by `dword << 4 | part`), the slots' mapping, the
        // meta mapping and where the landing has read up to now.
        let (differs, pushed, push) = {
            let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let differs = work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize));
            let mut pushed: Vec<crate::fx::Map<u32, u32>> = vec![Default::default(); plans.len()];
            let mut push_at = None;
            if let Some(push) = work.as_mut().and_then(|w| w.push.as_mut()) {
                let meta = push.meta.1;
                for (plan, map) in plans.iter().zip(pushed.iter_mut()) {
                    let Some((tag, _)) = plan.push else {
                        continue;
                    };
                    // SAFETY: inside the meta mapping; the plan's pass has
                    // run (its flight landed), so its count and records
                    // are written, and no later pass takes these slots
                    // before the landing says it has read them.
                    unsafe {
                        let count = ((meta + (PUSH_COUNTS + 4 * (tag % PUSH_TAGS)) as usize) as *const u32).read_volatile();
                        // The plan's slots follow the plans before it
                        // (passes run in the order of the queue). One
                        // that found no free slot, or another plan's
                        // record, is fetched instead.
                        for i in 0..count.min(PUSH_SLOTS) {
                            let slot = push.cursor.wrapping_add(i) & (PUSH_SLOTS - 1);
                            let record = (meta + (PUSH_RECORDS + 8 * slot) as usize) as *const u32;
                            if record.read_volatile() == tag {
                                map.insert(record.add(1).read_volatile(), slot);
                            }
                        }
                        push.cursor = push.cursor.wrapping_add(count);
                    }
                }
                push_at = Some((push.slots.1, meta, push.cursor));
            }
            (differs, pushed, push_at)
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        let mut title = Vec::new();
        let mut written = 0;
        if let Some((differs, dwords)) = differs {
            // SAFETY: the mapping of the results. The compares of these
            // plans have run, and their dwords are given to no other plan
            // before these are done with (`card_done`).
            let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
            let mut seen = crate::fx::Set::default();
            // The parts that went down with each plan's own pass: (range,
            // offsets in the copy, where they lie in the slots' mapping).
            let mut parts: Vec<Vec<(usize, usize, usize, usize)>> = vec![Vec::new(); plans.len()];
            if let Some((slots, _, _)) = push {
                for ((plan, map), parts) in plans.iter().zip(&pushed).zip(parts.iter_mut()) {
                    if plan.push.is_none() || map.is_empty() {
                        continue;
                    }
                    for (i, r) in plan.ranges.iter().enumerate().filter(|(_, r)| r.compared) {
                        for p in 0..r.span.pieces {
                            let dword = r.at + p;
                            let mask = differs.get(dword as usize).copied().unwrap_or(0) & 0xFFFF;
                            for part in (0..crate::shadow::CARD_PARTS).filter(|j| mask & (1 << j) != 0) {
                                let Some(&slot) = map.get(&(dword << 4 | part)) else {
                                    continue;
                                };
                                for (lo, hi) in r.shadow.card_piece_parts(&r.span, p, 1 << part) {
                                    // Where the pass started copying (as
                                    // `CARD_PUSH` takes it).
                                    let w = (r.span.first + p) as usize * 65536 + part as usize * 4096;
                                    let base = w.saturating_sub(r.span.skew as usize) & !3;
                                    if lo < base || hi - base > PUSH_SLOT as usize {
                                        continue;
                                    }
                                    if crate::census::skip_back(r.declared, (hi - lo) as u64) {
                                        continue;
                                    }
                                    parts.push((i, lo, hi, slots + slot as usize * PUSH_SLOT as usize + (lo - base)));
                                }
                            }
                        }
                    }
                }
            }
            // A part several plans of this landing pushed is merged from the
            // last of them only: its bytes hold what the earlier ones wrote
            // too (unless it wrote them over), as a fetch after all of them
            // did. Run C's flights rewrote the same parts again and again:
            // merged from every plan, they were 294 MB a frame.
            let mut last: crate::fx::Map<(usize, usize, usize), usize> = crate::fx::Map::default();
            let mut runs: Vec<PushedRun> = Vec::new();
            for (n, (plan, parts)) in plans.iter().zip(&parts).enumerate().rev() {
                for &(i, lo, hi, _) in parts {
                    last.entry((std::sync::Arc::as_ptr(&plan.ranges[i].shadow) as usize, lo, hi)).or_insert(n);
                }
            }
            for (n, (plan, parts)) in plans.iter().zip(&parts).enumerate() {
                // Merged from their slots, as they were when the plan's lists
                // had run (behind its own send of title bytes, ahead of later
                // ones); a part a later plan pushed is left to that plan.
                let mut merged = crate::fx::Set::default();
                if let Some((_, sent_as)) = plan.push {
                    // Parts that touch are merged as one run: they lie in
                    // the same pieces, and a run is one title memory read
                    // and one write instead of one of each per 4 KiB.
                    // Noted here and merged once every plan's are
                    // ([`merge_pushed`]), before the slots are given back.
                    let mut run: Vec<(usize, usize)> = Vec::new();
                    let mut run_at: Option<(&std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = None;
                    let merge = |run: &mut Vec<(usize, usize)>, run_at: &mut Option<(&std::sync::Arc<crate::shadow::Shadow>, usize, usize)>, runs: &mut Vec<PushedRun>| {
                        let Some((shadow, lo, _)) = run_at.take() else {
                            return;
                        };
                        runs.push((shadow.clone(), lo, std::mem::take(run), sent_as));
                    };
                    for &(i, lo, hi, at) in parts {
                        let shadow = &plan.ranges[i].shadow;
                        let key = (std::sync::Arc::as_ptr(shadow) as usize, lo, hi);
                        merged.insert(key);
                        if last.get(&key) != Some(&n) {
                            PUSH_STATS[3].fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        match run_at {
                            // `Shadow::card_merge_relay` takes a run of
                            // CARD_PARTS parts at most.
                            Some((before, _, end))
                                if std::sync::Arc::ptr_eq(before, shadow)
                                    && end == lo
                                    && run.len() < crate::shadow::CARD_PARTS as usize => {}
                            _ => {
                                merge(&mut run, &mut run_at, &mut runs);
                                run_at = Some((shadow, lo, lo));
                            }
                        }
                        if let Some((_, _, end)) = &mut run_at {
                            *end = hi;
                        }
                        // Inside the slots' mapping (checked when the part
                        // was taken).
                        run.push((at, hi - lo));
                        PUSH_STATS[0].fetch_add(1, Ordering::Relaxed);
                        PUSH_STATS[1].fetch_add((hi - lo) as u64, Ordering::Relaxed);
                    }
                    merge(&mut run, &mut run_at, &mut runs);
                }
                let flags: Vec<(usize, u32, u32)> = plan
                    .ranges
                    .iter()
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let mask = if r.compared { differs.get((r.at + p) as usize).copied().unwrap_or(u32::MAX) } else { u32::MAX };
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, mask)
                        })
                    })
                    .collect();
                let fetched = flagged_parts(&flags);
                for r in &plan.ranges {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let Some(&mask) = fetched.get(&(copy, r.span.first + p)) else {
                            continue;
                        };
                        // Only the 4 KiB parts flagged, each once; touching
                        // parts are fetched and merged as one.
                        for (lo, hi) in r.shadow.card_piece_parts(&r.span, p, mask) {
                            if merged.contains(&(copy, lo, hi))
                                || crate::census::skip_back(r.declared, (hi - lo) as u64)
                                || !seen.insert((copy, lo, hi))
                            {
                                continue;
                            }
                            if plan.push.is_some() {
                                PUSH_STATS[2].fetch_add(1, Ordering::Relaxed);
                            }
                            match pieces.last_mut() {
                                Some(last) if std::sync::Arc::ptr_eq(&last.0, &r.shadow) && last.2 == lo => last.2 = hi,
                                _ => pieces.push((r.shadow.clone(), lo, hi, total)),
                            }
                            total += (hi - lo) as u64;
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                    crate::census::compared(r.shadow.title_start(), u64::from(r.span.pieces));
                }
            }
            written += merge_pushed(runs);
        }
        // The pushed parts are read: their slots may be taken again. Later
        // merges need no send older than the last plan's own (a plan's
        // pushed parts were taken behind its send, so sends up to it are
        // no use to it; any fetch below is later still).
        if let Some((_, meta, cursor)) = push {
            // SAFETY: inside the meta mapping; the card only reads it.
            unsafe { ((meta + 4) as *mut u32).write_volatile(cursor) };
        }
        let pushed_after = plans.iter().filter_map(|p| p.push.map(|(_, sent_as)| sent_as)).max();
        let fetched = |after: u64| crate::shadow::relay_fetched(pushed_after.map_or(after, |p| p.min(after)));
        if pieces.is_empty() {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            // Nothing to fetch: every send so far is older than the next fetch.
            fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
            return;
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        if crate::census::on() {
            for (shadow, lo, hi, _) in &pieces {
                crate::census::fetched(shadow.title_start(), (hi - lo) as u64);
            }
        }
        for (shadow, lo, hi, at) in &pieces {
            // SAFETY: the mapping of the fetched bytes, `total` of them;
            // the copy into it has run.
            let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
            let merged = shadow.card_merge_relay(*lo, &[gpu], after, &mut title);
            shadow.walls_merged(gpu.len(), merged);
            written += merged;
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        // Sends up to `after` are of no use to any later merge.
        fetched(after);
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<&CardPlan> = flights.iter().filter_map(|f| f.card.as_ref()).collect();
                self.relay_card_back(&plans, fetch);
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                relay_hand_back(&back);
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => {
                    // Out of `in_hand` before the title can see it: a reset
                    // and a new wait must not match a write that has landed
                    // (fxceless #785).
                    {
                        let mut line = line();
                        if let Some(pos) =
                            line.in_hand.iter().position(|&(a, _, b)| a == at && b == bytes)
                        {
                            line.in_hand.remove(pos);
                        }
                    }
                    what.write(at as *mut u8, bytes);
                }
                Late::Signal(fence, value) => crate::fence::set(fence.as_this(), value),
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        self.log_messages();
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                by_copy,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                landing,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            landing.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            by_copy.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            let mut copy_ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    for (shadow, r) in &flight.back {
                        let owner = std::sync::Arc::as_ptr(shadow) as usize;
                        ranges.extend(r.iter().copied());
                        copy_ranges.extend(r.iter().map(|&(a, b)| (owner, a, b)));
                    }
                    ranges.extend(flight.backs.iter().copied());
                    copy_ranges.extend(flight.backs.iter().map(|&(a, b)| (0, a, b)));
                }
            }
            join_ranges(written, ranges);
            join_copy_ranges(by_copy, copy_ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                let mask = if r.compared { differs[(r.at + p) as usize] } else { u32::MAX };
                if mask == 0 {
                    continue;
                }
                // Only the 4 KiB parts the compare flagged, each once;
                // touching parts are fetched and merged as one.
                for (lo, hi) in r.shadow.card_piece_parts(&r.span, p, mask) {
                    if crate::census::skip_back(r.declared, (hi - lo) as u64)
                        || !seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi))
                    {
                        continue;
                    }
                    match pieces.last_mut() {
                        Some(last) if last.0 == i && last.2 == lo => last.2 = hi,
                        _ => pieces.push((i, lo, hi, total)),
                    }
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            crate::census::compared(r.shadow.title_start(), u64::from(r.span.pieces));
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if crate::census::on() {
                for &(i, lo, hi, _) in &pieces {
                    crate::census::fetched(plan.ranges[i].shadow.title_start(), (hi - lo) as u64);
                }
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        let _ = self.submit(runs, false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, by_copy, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.backs.iter().copied()),
            );
            join_copy_ranges(
                by_copy,
                sent.back
                    .iter()
                    .flat_map(|(shadow, ranges)| {
                        let owner = std::sync::Arc::as_ptr(shadow) as usize;
                        ranges.iter().map(move |&(a, b)| (owner, a, b))
                    })
                    .chain(sent.backs.iter().map(|&(a, b)| (0, a, b))),
            );
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                backs: sent.backs,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, by_copy, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.backs.iter().copied()),
            );
            join_copy_ranges(
                by_copy,
                sent.back
                    .iter()
                    .flat_map(|(shadow, ranges)| {
                        let owner = std::sync::Arc::as_ptr(shadow) as usize;
                        ranges.iter().map(move |&(a, b)| (owner, a, b))
                    })
                    .chain(sent.backs.iter().map(|&(a, b)| (0, a, b))),
            );
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                backs: sent.backs,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for a direct queue fence with the shared event, under the submission lock.
    fn wait_for_submission(&self, target: u64, started: std::time::Instant, context: &str) {
        // SAFETY: the fence and event are ours, and the submission lock
        // prevents another wait from using this event.
        unsafe {
            if let Err(e) = self.fence.SetEventOnCompletion(target, self.event) {
                fail(format_args!("could not register a direct fence wait for {context} ({e})"));
            }
            working(false);
            match WaitForSingleObject(self.event, INFINITE) {
                WAIT_OBJECT_0 => {}
                WAIT_FAILED => {
                    let error = windows::core::Error::from_thread();
                    fail(format_args!("could not wait for {context} ({error})"));
                }
                status => fail(format_args!("unexpected direct fence wait result for {context} ({:#x})", status.0)),
            }
            stood(started);
            working(true);
        }
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.by_copy.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                self.wait_for_submission(*submitted, waiting, "submissions sent ahead");
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                hand_back(&back);
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence.as_this(), value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// A detile into a texture's staging buffer waits until submission
    /// `target` (sent ahead, whose prelude copies out of that buffer) has
    /// run: otherwise the texture takes part of the newer contents.
    pub(crate) fn wait_for_staging(&self, target: u64) {
        // SAFETY: the fence belongs to this host.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            fail(format_args!("device removed while waiting to reuse texture staging"));
        }
        if completed >= target {
            return;
        }
        let started = std::time::Instant::now();
        let was_working = working(false);
        // SAFETY: `target` was signalled on the host queue when it was sent;
        // a null event waits inline.
        let waited = unsafe { self.fence.SetEventOnCompletion(target, HANDLE::default()) };
        working(was_working);
        if let Err(error) = waited {
            fail(format_args!("could not wait to reuse texture staging ({error})"));
        }
        // SAFETY: a plain query; device removal can also end the wait.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX || completed < target {
            fail(format_args!("texture staging upload did not reach fence {target}"));
        }
        stood(started);
        STAGING_WAIT_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        // D3D12 forbids a command list twice in one `ExecuteCommandLists`
        // (drivers hang: DXGI_ERROR_DEVICE_HUNG), and a list's second
        // `prelude` would reset the one queued for its first occurrence.
        // Split at a duplicate: the part before it runs (and is waited for)
        // first (fxceless/victory-fork#675).
        if let Some(split) = first_duplicate(runs) {
            let _ = self.submit(&runs[..split], false);
            return self.submit(&runs[split..], late);
        }
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        submitting(1, runs.len());
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        // Refreshes over what another copy's shaders write in a flight take
        // those bytes on the card ([`alias_forward`]) and need not wait.
        let (mut forwards, forwarded) = if late { plan_forwards(runs) } else { (Vec::new(), Vec::new()) };
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = late_blocked(runs, &forwarded) {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| {
                    let immed = line.immed_writes();
                    blocked_by(&line.written, &line.by_copy, &immed, runs, false, &forwarded).is_none()
                });
            } else {
                // All landed: title memory has the flights' bytes, and the
                // refreshes take them from there.
                self.settle(&mut submitted, why);
                forwards.clear();
            }
        }
        // The copies between host copies the card has passed are forgotten;
        // around those still queued the lists are made ready
        // ([`handed_ready`]), or the flights settled where they cannot be.
        self.handed_passed();
        let mut writers = Vec::new();
        if late {
            match handed_ready(runs, &forwards) {
                Ok(after) => writers = after,
                Err(()) => {
                    FORWARD_STATS[4].fetch_add(1, Ordering::Relaxed);
                    self.settle(&mut submitted, WHY_READ);
                    forwards.clear();
                    self.handed_passed();
                }
            }
        }
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            // All landed: the refreshes take title memory.
            forwards.clear();
            self.handed_passed();
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        crate::descriptor::make_waiting();
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            crate::watch::ask_once(&ranges);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_under_flights(runs) {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                forwards.clear();
                self.handed_passed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && stale_under_flights(runs) {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
            forwards.clear();
            self.handed_passed();
        }
        // From here the CPU leaves the ranges alone that take their bytes on
        // the card, until the card has passed the copies (sent below, ahead
        // of this submission's lists).
        hand(&forwards, *submitted + 1);
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.fill_now();
        batch.flush();
        crate::watch::ask_once_end();
        // The writing copies of ranges the CPU took back while their copy
        // on the card is still queued ([`handed_ready`]): brought up to
        // date after the refreshing copies, so that the queued copy brings
        // nothing older than those now hold.
        for (writer, a, b) in writers {
            writer.update(a, b - a);
        }
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        let card = self.card_before(runs, flown);
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        // The in-flight bytes the refreshes did not wait for, copied on the
        // card ahead of the lists that read them (the refreshes on the CPU
        // are done: these copies go over what they put there).
        if !forwards.is_empty() && !self.send_forwards(&forwards, *submitted + 1) {
            // No list to record them in: the flights land, and the bytes
            // go from copy to copy on the CPU (the refreshes ran before).
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_READ);
            for f in &forwards {
                f.to.take_from(&f.from, f.address, f.end);
            }
            // Nothing is queued: the ranges are the CPU's again.
            let never = *submitted + 1;
            handed_drop(|h| h.done == never);
            self.handed_passed();
            settled += settling.elapsed();
        }
        let prepared = std::time::Instant::now();
        if late {
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            // Staging buffers these lists' preludes copy out of: a detile
            // into one waits for this submission (PalmVIDev fc836695).
            for list in runs {
                for (_, staging) in &list.deferred().texture_uploads {
                    staging.upload_sent(Some(*submitted));
                }
            }
            let back = declared_writes(runs);
            let backs: Vec<(usize, usize)> = runs
                .iter()
                .flat_map(|l| {
                    l.deferred()
                        .backs
                        .iter()
                        .filter(|&&(_, _, [row, rows, _])| row > 0 && rows > 0)
                        .map(|&(_, at, [row, rows, pitch])| {
                            let span = rows.saturating_sub(1).saturating_mul(pitch).saturating_add(row);
                            (at, at.saturating_add(span))
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                backs,
                writes,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        // Waited for below, under the submission lock: no detile can come
        // between, so nothing is left to wait for.
        for list in runs {
            for (_, staging) in &list.deferred().texture_uploads {
                staging.upload_sent(None);
            }
        }
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                if let Err(e) = self.queue.Signal(&self.fence, *submitted) {
                    fail(format_args!("could not signal a synchronous host submission ({e})"));
                }
            }
            self.wait_for_submission(*submitted, started, "a synchronous host submission");
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = declared_writes(runs);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded,
    /// or `None` if the host resource lacks `ALLOW_RENDER_TARGET`.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        if let Some(res) = resource {
            // SAFETY: a live resource.
            let flags = unsafe { res.GetDesc() }.Flags;
            if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 == 0 {
                return None;
            }
        }
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            Some(handle)
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        // Hoisted clamps it holds: readable as arguments by the list after.
        self.clamp_end();
        self.twins_to_depth();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        *self.deferred() = Deferred::default();
        self.bump_hoist_mark();
        {
            let mut hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
            (hoisted.used, hoisted.at) = (0, 0);
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.buffer_states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        if let Some(c) = self.card_constants.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // Every submission of the list has run: the buffer is free, and
            // in the common state again.
            (c.1, c.2) = (0, false);
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Moves a `Shadow` buffer (created in `COMMON`) to `wanted`
    /// (`GENERIC_READ` or `UNORDERED_ACCESS`) in this list and records the
    /// transition barrier when its state changes. Buffers always decay back
    /// to `COMMON` at the end of `ExecuteCommandLists`, so no prelude is
    /// needed.
    pub(crate) fn buffer_state(&self, buffer: &ID3D12Resource, wanted: D3D12_RESOURCE_STATES) {
        if wanted == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            // SAFETY: a live buffer.
            let flags = unsafe { buffer.GetDesc() }.Flags;
            if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
                return;
            }
        }
        let before = self
            .buffer_states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .transition_for(buffer, wanted);
        if let Some(before) = before {
            let barrier = transition(buffer, before, wanted);
            // SAFETY: an open direct command list and a live buffer.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// Transitions a `Shadow` buffer to `GENERIC_READ` (VB, IB, CBV, SRV,
    /// indirect argument) if it is not already there in this list.
    pub(crate) fn buffer_read(&self, buffer: &ID3D12Resource) {
        self.buffer_state(buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    /// Transitions a `Shadow` buffer to `UNORDERED_ACCESS` if it is not
    /// already there in this list.
    pub(crate) fn buffer_uav(&self, buffer: &ID3D12Resource) {
        self.buffer_state(buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind. For
    /// title-written textures whose staging buffer was refreshed (or has
    /// not been uploaded yet) it copies the staging buffer into the host
    /// texture before the list uses it.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        // `run_deferred` has already refreshed each staging buffer, so we
        // know here exactly which textures need a GPU copy on this submission.
        let uploads: Vec<_> = {
            let deferred = self.deferred();
            deferred
                .texture_uploads
                .iter()
                .filter_map(|(copy, staging)| {
                    if staging.take_upload() {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::TexUploads, 1);
                        // A later detile waits for this copy if the
                        // submission is sent ahead (`Staging::upload_sent`).
                        staging.upload_queued();
                        Some((copy.clone(), staging.clone()))
                    } else {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
                        None
                    }
                })
                .collect()
        };
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        for (copy, _) in &uploads {
            let key = std::sync::Arc::as_ptr(copy) as usize;
            if copying.contains_key(&key) {
                continue;
            }
            let real = copy.state.load(Ordering::Relaxed);
            copying.insert(key, COPY_DEST);
            if real != COPY_DEST {
                to_copy.push(transition(
                    &copy.texture,
                    host_state(real),
                    host_state(COPY_DEST),
                ));
                if !states.contains_key(&key) {
                    barriers.push(transition(
                        &copy.texture,
                        host_state(COPY_DEST),
                        host_state(real),
                    ));
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if barriers.is_empty() && syncs.is_empty() && uploads.is_empty() {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                for (copy, staging) in &uploads {
                    staging.record_copy(list, &copy.texture);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.texture_uploads.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        self.bump_hoist_mark();
        deferred.refreshed.clear();
        deferred.memory_read_seen = 0;
    }

    /// With [`fill_on_card`]: `len` bytes of the host copy `shadow` from title
    /// address `address`, copied on the card where the list stands into its
    /// device-local constants buffer: the GPU address of the bytes there (on
    /// 256 bytes, for a constant buffer view), or `None` (no buffer, no room).
    ///
    /// A fill of upload memory reads title memory on the CPU at submission,
    /// and a submission whose fill reads what a flight's shaders write waits
    /// for the flight to land (52-80 ms a frame in the world, one table
    /// constant buffer of 512 bytes). Read on the card, behind the flight on
    /// the same queue, those bytes are what the shaders wrote without the
    /// wait; title writes to them reach the copy first, through its refresh.
    pub(crate) fn card_constants(&self, shadow: &crate::shadow::Shadow, address: u64, len: u64) -> Option<u64> {
        const SIZE: u64 = 1 << 20;
        let host = host()?;
        let offset = shadow.offset_in(address)?;
        if len == 0 || shadow.left(address) < len {
            return None;
        }
        let room = len.next_multiple_of(256);
        let mut c = self.card_constants.lock().unwrap_or_else(|e| e.into_inner());
        if c.is_none() {
            let buffer = host.uav_buffer(SIZE)?;
            name(&buffer, || "list constants on the card".into());
            *c = Some((buffer, 0, false));
        }
        let (buffer, used, readable) = c.as_mut()?;
        let buffer: &ID3D12Resource = buffer;
        if *used + room > SIZE {
            return None;
        }
        let at = *used;
        *used += room;
        let constants = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        // SAFETY: an open list; both buffers are live and the ranges inside
        // them (checked above).
        unsafe {
            if *readable {
                self.list.ResourceBarrier(&[transition(buffer, constants, D3D12_RESOURCE_STATE_COPY_DEST)]);
            }
            // What shaders of this list wrote to the copy before, first.
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
            self.list.CopyBufferRegion(buffer, at, &shadow.buffer, offset, len);
            self.list.ResourceBarrier(&[transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST, constants)]);
            *readable = true;
            Some(buffer.GetGPUVirtualAddress() + at)
        }
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        {
            let mut deferred = self.deferred();
            if deferred.quarantined {
                return;
            }
            deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
        }
        self.bump_hoist_mark();
    }

    /// The list records a quarantined stage of the structure builder
    /// (`VICTORY_BVH_QUARANTINE`, `VICTORY_BVH_WARMUP`): no shader write of
    /// the whole list is copied back into title memory, those declared
    /// before it included.
    pub(crate) fn quarantine(&self) {
        let mut deferred = self.deferred();
        if !deferred.quarantined {
            deferred.quarantined = true;
            deferred.copies.retain(|step| !matches!(step, Step::GpuWrite(..)));
        }
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// Whether shaders of this list, as recorded so far, may write title
    /// memory in `[address, address + len)` (a range declared with
    /// [`HostList::defer_gpu_write`] touches it).
    pub(crate) fn gpu_wrote(&self, address: usize, len: usize) -> bool {
        let end = address.saturating_add(len);
        self.deferred().copies.iter().any(|step| {
            matches!(step, Step::GpuWrite(_, at, n) if *at < end && address < at.saturating_add(*n))
        })
    }

    /// For a copy out of title buffer memory that this list's shaders
    /// write (a dispatch fills a buffer, the list then copies the buffer
    /// into a texture): a host buffer holding `[address, address + len)`
    /// as it is at this point of the list, in the copy source state. The
    /// bytes are copied on the GPU from the memory's host copy, after the
    /// dispatches recorded before; the title's memory only gets them once
    /// the list has run, so a copy filled from it at submission would take
    /// what was there before the list. `None` if no shader of the list
    /// writes the range, or it has no host copy: the caller then copies
    /// from title memory as before. (PalmVIDev 4d1ce258.)
    pub(crate) fn staged_written_source(&self, host: &Host, address: u64, len: u64) -> Option<ID3D12Resource> {
        if len == 0 || !self.gpu_wrote(address as usize, usize::try_from(len).ok()?) {
            return None;
        }
        let shadow = crate::shadow::find(address, len)?;
        let offset = shadow.offset_in(address)?;
        if shadow.left(address) < len {
            return None;
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut staged: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut staged,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, len, "no buffer to stage a shader-written copy source");
            return None;
        }
        let staged = staged?;
        name(&staged, || format!("shader-written copy source {address:#x}+{len:#x}"));
        // What the title wrote there itself reaches the host copy at
        // submission, as for a shader that reads the range.
        shadow.refresh(self, address, len);
        // SAFETY: an open list; the range lies in the host copy (checked
        // above) and the staged buffer holds `len` bytes.
        unsafe {
            // What shaders of this list wrote to the copy before, first.
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
            self.list.CopyBufferRegion(&staged, 0, &shadow.buffer, offset, len);
            self.list.ResourceBarrier(&[transition(
                &staged,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        self.keep(&staged);
        STAGED_WRITTEN.fetch_add(1, Ordering::Relaxed);
        Some(staged)
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().hold_upload(buffer, len);
            return Some((gpu, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, gpu) = (s.chunks[s.current].0.data, s.chunks[s.current].0.gpu);
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all): taken, then zeroed.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                for k in 0..3 {
                    words.add(k).write_volatile(0);
                }
            })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Advances the list's title-memory copy watermarks to `total` (`writes`:
    /// whether this dispatch may write title memory by address) and returns
    /// `(read_from, write_from)`: the first copy index still needing a
    /// refresh and, when `writes`, the first still needing a write-back
    /// declaration.
    pub(crate) fn memory_sync_range(&self, total: usize, writes: bool) -> (usize, usize) {
        self.deferred().memory_sync_range(total, writes)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.bump_hoist_mark();
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.memory_read_seen = 0;
    }

    /// What was recorded so far that may change indirect arguments, as a
    /// mark: clamps hoisted before this list (`crate::indirect::hoisting`)
    /// stay good for its indirect calls while it is the same.
    pub(crate) fn hoist_mark(&self) -> u64 {
        self.hoist_mark.load(Ordering::Relaxed)
    }

    /// Something recorded may change indirect arguments (or a hoist begins
    /// on this list): a mark no list had before.
    pub(crate) fn bump_hoist_mark(&self) {
        let next = HOIST_MARKS.fetch_add(1, Ordering::Relaxed);
        self.hoist_mark.store(next, Ordering::Relaxed);
    }

    /// Room for `bytes` of hoisted clamped indirect arguments in this
    /// list's buffers, writable by shaders from here on in the list: the
    /// buffer and the offset (on 16 bytes). None for more than a buffer
    /// holds. The first room of a recording waits for every earlier shader
    /// write (the title's passes that wrote the arguments).
    pub(crate) fn clamp_room(&self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        let bytes = bytes.max(16).next_multiple_of(16);
        if bytes > HOIST_CHUNK {
            return None;
        }
        let mut h = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        if h.used > 0 && h.at + bytes <= HOIST_CHUNK {
            let at = h.at;
            h.at += bytes;
            return Some((h.chunks[h.used - 1].clone(), at));
        }
        if h.used == h.chunks.len() {
            let buffer = host()?.uav_buffer(HOIST_CHUNK)?;
            name(&buffer, || "hoisted indirect arguments".to_owned());
            h.chunks.push(buffer);
        }
        let first = h.used == 0;
        let buffer = h.chunks[h.used].clone();
        h.used += 1;
        h.at = bytes;
        let mut barriers = vec![transition(
            &buffer,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )];
        if first {
            barriers.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            });
        }
        // SAFETY: an open list and a live buffer of its own, common at the
        // start of every recording.
        unsafe { self.list.ResourceBarrier(&barriers) };
        Some((buffer, 0))
    }

    /// The hoisted clamped arguments written in this recording become
    /// readable as indirect arguments by the lists after it ([`close`]).
    ///
    /// [`close`]: HostList::close
    pub(crate) fn clamp_end(&self) {
        let used: Vec<ID3D12Resource> = {
            let h = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
            h.chunks[..h.used].to_vec()
        };
        if used.is_empty() {
            return;
        }
        let barriers: Vec<_> = used
            .iter()
            .map(|b| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )
            })
            .collect();
        // SAFETY: an open list and live buffers it left for unordered access.
        unsafe { self.list.ResourceBarrier(&barriers) };
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Copies `staging` into `copy` in this list's prelude when submitted,
    /// if `staging` has not been uploaded yet or was detiled by `refresh`.
    pub(crate) fn defer_texture_upload(
        &self,
        copy: std::sync::Arc<crate::resource::HostTexture>,
        staging: std::sync::Arc<crate::texture::Staging>,
    ) {
        self.deferred().texture_uploads.push((copy, staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let deferred = self.deferred();
        let mut steps = deferred.copies.iter().peekable();
        while let Some(step) = steps.next() {
            let started;
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    started = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    len
                }
                &Step::Fill(dst, src, len) if fills_joined() => {
                    // They wait in the submission's batch until a step that
                    // writes title memory, and go together (`fills_joined`).
                    batch.fills.push((dst, src, len));
                    while let Some(&Step::Fill(dst, src, len)) = steps.next_if(|s| matches!(s, Step::Fill(..))) {
                        batch.fills.push((dst, src, len));
                    }
                    continue;
                }
                &Step::Fill(dst, src, len) => {
                    started = std::time::Instant::now();
                    let fill = |dst: usize, src: usize, len: usize| {
                        #[cfg(feature = "diagnostics")]
                        {
                            add(C::Copies, 1);
                            add(C::CopyBytes, len as u64);
                        }
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        len
                    };
                    let mut copied = fill(dst, src, len);
                    // A run of fills (a list's constant buffers, mostly a
                    // few hundred bytes each) is timed as one: reading the
                    // clock around each would cost as much as the copies.
                    while let Some(&Step::Fill(dst, src, len)) =
                        steps.next_if(|s| matches!(s, Step::Fill(..)))
                    {
                        copied += fill(dst, src, len);
                    }
                    copied
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    #[cfg(feature = "diagnostics")]
                    let run = std::time::Instant::now();
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, run.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=0` disables it; `=1` forces it).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_refresh_enabled(
            std::env::var("VICTORY_WIDE_REFRESH").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_refresh_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into
/// disjoint ranges ([`push_range`]): ranges that overlap or lie within a
/// page of each other join, distant ranges of one copy are brought up apart
/// instead of the span between them (upstream #2428). Done before a step
/// that writes title memory or gives a range to shaders, and before the
/// lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: crate::fx::Map<usize, usize>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: crate::fx::Set<usize>,
    /// Fills of upload memory not done yet ([`fills_joined`]): done before
    /// a step that writes title memory and at the submission's end.
    fills: Vec<(usize, usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        match self.index.get(&key) {
            Some(&at) => push_range(&mut self.refreshes[at].1, address, end),
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), vec![(address, end)]));
            }
        }
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let (shadow, ranges) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        for (from, to) in ranges {
            Self::update(&shadow, from, to);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.fill_now();
        self.flush();
        self.textures.clear();
    }

    /// The fills waiting ([`fills_joined`]), done now.
    fn fill_now(&mut self) {
        if !self.fills.is_empty() {
            fill_joined(&mut self.fills);
        }
    }

    fn flush(&mut self) {
        self.index.clear();
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        let ahead = self.refreshes.len() > 1;
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|r| !r.0.unwatched())
                .flat_map(|r| r.1.iter().copied())
                .collect();
            crate::watch::ask_ahead(&ranges);
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if wide_refresh() && self.refreshes.len() >= 4 {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                .refreshes
                .drain(..)
                .map(|(shadow, ranges)| {
                    Box::new(move || {
                        for (from, to) in ranges {
                            Self::update(&shadow, from, to);
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            PREPARING[0].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
        } else {
            for (shadow, ranges) in self.refreshes.drain(..) {
                for (from, to) in ranges {
                    Self::update(&shadow, from, to);
                }
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Adds `[address, end)` to a copy's refresh ranges, kept disjoint and in
/// order: what it overlaps, or lies within a page of, joins it (one copy
/// is cheaper than two across a page's gap), and ranges it bridges join
/// together. Distant ranges stay apart, so the span between them is not
/// copied. From upstream #2428 (borosf).
fn push_range(ranges: &mut Vec<(usize, usize)>, address: usize, end: usize) {
    const GAP: usize = crate::watch::PAGE;
    let (mut from, mut to) = (address, end);
    ranges.retain(|&(a, e)| {
        let joins = a <= to.saturating_add(GAP) && from <= e.saturating_add(GAP);
        if joins {
            (from, to) = (from.min(a), to.max(e));
        }
        !joins
    });
    let at = ranges.partition_point(|&(a, _)| a < from);
    ranges.insert(at, (from, to));
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// Both ways of reading the source report an unmapped page instead of
/// faulting: in user mode into host memory ([`crate::title_copy`]), through
/// the kernel (`ReadProcessMemory`) into title memory. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // A page that cannot be read fails either way and is found below.
    let whole = if title_dst {
        read(src, dst, len)
    } else {
        // SAFETY: the destination is host memory of `len` bytes.
        unsafe { crate::title_copy::copy(dst, src, len) }
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// The end of the run of committed, readable (or writable) pages that
/// starts at `at`, cut at `end`. `None` if the page at `at` is not one.
///
/// The kernel calls cost 40 MB/s reading and 10 MB/s writing here: they walk
/// the range page by page. One `VirtualQuery` per region is enough to know
/// the bytes are there, and the copy is then memory speed. Title memory is
/// view memory with holes, so this answers per run, not per range.
fn mapped_run(at: usize, end: usize, write: bool) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at;
    while p < end {
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(p as _, &mut info, size_of_val(&info)) };
        if size != size_of_val(&info) {
            return (p > at).then_some(p);
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return (p > at).then_some(p);
        }
        let protect = info.Protect & 0xff;
        let writable = matches!(protect, 0x04 | 0x08 | 0x40 | 0x80);
        if info.State != MEM_COMMIT
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0
            || (write && !writable)
        {
            return (p > at).then_some(p);
        }
        p = next;
    }
    Some(end)
}

fn mapped_whole(at: usize, len: usize, write: bool) -> bool {
    mapped_run(at, at.saturating_add(len), write) == Some(at.saturating_add(len))
}

/// Whether a submission's fills of upload memory from title memory are read
/// together (on by default; `VICTORY_JOIN_FILLS=0` turns it off): those
/// between two steps that write title memory, from all the submission's
/// lists, sorted by where they read; the ones less than 4 KiB apart go by
/// one kernel read (spans of 1 MiB at most) and are handed out from there.
/// Each fill is a kernel read of its own otherwise ([`copy_mapped`]): some
/// 12,000 a world frame, mostly constants a few hundred bytes long.
/// PalmVIDev #229 and #245 (EricPlayZ: submission-thread copies 37.5 ->
/// 31.4 ms a frame on an RTX 2070 SUPER).
fn fills_joined() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_FILLS").is_none_or(|v| v != "0"))
}

/// Does `fills` (destination, source, bytes) as [`fills_joined`] says and
/// empties it. A span the kernel cannot read whole (a page not mapped) goes
/// fill by fill, as without the switch.
fn fill_joined(fills: &mut Vec<(usize, usize, usize)>) {
    const GAP: usize = 4 << 10;
    const MOST: usize = 1 << 20;
    thread_local! {
        static SPAN: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    fills.sort_unstable_by_key(|f| f.1);
    let mut i = 0;
    while i < fills.len() {
        let (from, mut to) = (fills[i].1, fills[i].1 + fills[i].2);
        let mut j = i + 1;
        while j < fills.len() && fills[j].1 <= to + GAP && fills[j].1 + fills[j].2 - from <= MOST {
            to = to.max(fills[j].1 + fills[j].2);
            j += 1;
        }
        let whole = SPAN.with(|span| {
            let mut span = span.borrow_mut();
            span.resize(to - from, 0);
            if !read(from as *const u8, span.as_mut_ptr(), to - from) {
                return false;
            }
            for &(dst, src, len) in &fills[i..j] {
                // SAFETY: the destination is a list's upload memory of `len`
                // bytes; the source lies in the span just read.
                unsafe { std::ptr::copy_nonoverlapping(span.as_ptr().add(src - from), dst as *mut u8, len) };
            }
            true
        });
        if !whole {
            for &(dst, src, len) in &fills[i..j] {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        }
        i = j;
    }
    fills.clear();
}

impl Drop for Batch {
    /// Fills still waiting are done, whoever ends the batch.
    fn drop(&mut self) {
        self.fill_now();
    }
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    // `VirtualQuery` walks the page tables to the region's end: ~0.6 ms in
    // the title's big pools, where most copies are under a kilobyte. Only a
    // large read is worth the question.
    if len == 0 {
        return true;
    }
    // The answer is only a hint: another thread may remap or decommit the
    // pages between the question and the copy (the title's own frees, our
    // splits and remaps of its views). A plain memcpy then faulted on a
    // reserved page; the copy is ours that reports a fault instead
    // ([`crate::title_copy`]), and the kernel's copy below takes over.
    // (Trexies cd41200 dropped this path for the race; this keeps its speed.)
    if len >= (256 << 10)
        && mapped_whole(src as usize, len, false)
        // SAFETY: `dst` is `len` bytes the caller may write; a fault on
        // either range returns false.
        && unsafe { crate::title_copy::copy(dst, src, len) }
    {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::MoveFast, len as u64);
        return true;
    }
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0usize;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source. `NtReadVirtualMemory` is the same kernel
    // copy `ReadProcessMemory` wraps, without the wrapper's validation of
    // both ranges (`VICTORY_COPY_MODE=plain` measured that wrapper as most
    // of a copy's cost here: 14k copies of 18 KiB at ~84 MB/s).
    let status = unsafe {
        NtReadVirtualMemory(
            GetCurrentProcess(),
            src.cast(),
            dst.cast(),
            len,
            &mut done,
        )
    };
    status >= 0 && done == len
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's read from a process's memory, `(process, address,
    /// bytes, length, read out)`: an NTSTATUS, negative when a page of the
    /// range is not there (a partial read is all it gives then).
    fn NtReadVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *const std::ffi::c_void,
        buffer: *mut std::ffi::c_void,
        size: usize,
        read: *mut usize,
    ) -> i32;

    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 {
        return;
    }
    // A small write tries the one kernel call first: `mapped_run`'s
    // `VirtualQuery` walks the page tables to the region's end (~0.6-2 ms in
    // the title's big pools), more than the call costs when it succeeds.
    // Callers such as `card_merge_relay` write many short stretches.
    if len < (256 << 10) && write(dst as usize, len) {
        return;
    }
    // Title memory is view memory with holes: copy the committed runs at
    // memory speed and give the holes to the kernel, one piece at a time.
    let mut at = dst as usize;
    let end = at + len;
    while at < end {
        let run = mapped_run(at, end, true);
        // `mapped_run`'s answer is only a hint: the title (or our own split
        // and remap of its views) may unmap or decommit the pages before the
        // copy. A plain memcpy then faulted on a reserved page and took the
        // process down; ours reports the fault ([`crate::title_copy`]) and
        // the run goes to the kernel below, which writes what is there.
        if let Some(to) = run
            // SAFETY: the source is our host copy of `len` bytes, the
            // destination title memory; a fault on either returns false.
            && unsafe {
                crate::title_copy::copy(at as *mut u8, src.wrapping_add(at - dst as usize), to - at)
            }
        {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MoveFast, (to - at) as u64);
            at = to;
            continue;
        }
        let step = 64 << 10;
        let to = match run {
            // the run changed under us: the kernel, a 64 KiB piece at a time
            Some(r) => ((at / step + 1) * step).min(r),
            None => ((at / step + 1) * step).min(end),
        };
        if !write(at, to - at) {
            for_pieces(at, to - at, 4 << 10, |at, n| {
                write(at, n);
            });
        }
        at = to;
    }
}

/// Our own write of title memory `[at, at + len)` (what shaders wrote, put
/// back): through a view of our own of the same memory when the guards hold
/// all of it read-only ([`crate::watch::aside`]), so the title's views stay
/// read-only and nothing is opened and guarded again; else the guards are
/// opened ([`crate::watch::writing`]) and the bytes go in by
/// [`write_mapped`], as before. The blocks are noted written when it drops,
/// after the bytes are in. In the run of main 7e88d16 about 1,000 of the
/// 1,500 blocks opened a frame were opened for our own writes, each a
/// `VirtualProtect` there and another at the next asking.
pub(crate) struct TitleWrite {
    at: usize,
    len: usize,
    aside: Option<*mut u8>,
}

impl TitleWrite {
    pub(crate) fn open(at: usize, len: usize) -> Self {
        let aside = crate::watch::aside(at, len);
        if aside.is_none() {
            crate::watch::writing(at, len);
        }
        TitleWrite { at, len, aside }
    }

    /// Writes `n` bytes from `src` at title address `dst`, which lies in the
    /// range opened (a write outside it opens its own).
    pub(crate) fn put(&self, dst: usize, src: *const u8, n: usize) {
        match self.aside {
            Some(base) if dst >= self.at && dst.saturating_add(n) <= self.at + self.len => {
                // SAFETY: our own view of the range's memory, all of it
                // committed (`watch::aside`); `src` holds `n` bytes of ours.
                unsafe { std::ptr::copy_nonoverlapping(src, base.add(dst - self.at), n) };
            }
            Some(_) => {
                crate::watch::writing(dst, n);
                write_mapped(dst as *mut u8, src, n);
            }
            None => write_mapped(dst as *mut u8, src, n),
        }
    }
}

impl Drop for TitleWrite {
    fn drop(&mut self) {
        if self.aside.is_some() {
            crate::watch::wrote_aside(self.at, self.len);
        }
    }
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        crate::census::made(0, size);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    /// A mapped upload buffer with room for `size` bytes
    /// ([`upload_room`]: the third value), zeroed like a new one: one given
    /// back with [`give_upload`] if there is one of that room, else made.
    /// The caller gives it back when nothing reads it any more.
    pub(crate) fn upload_pooled(&self, size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
        let room = upload_room(size);
        let kept = {
            let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
            let at = pool.kept.iter().rposition(|&(_, _, r)| r == room);
            at.and_then(|at| pool.kept.remove(at)).inspect(|_| pool.bytes -= room)
        };
        if let Some((buffer, data, _)) = kept {
            UPLOADS_TAKEN[0].fetch_add(1, Ordering::Relaxed);
            UPLOADS_TAKEN[1].fetch_add(room, Ordering::Relaxed);
            let data = data as *mut u8;
            // SAFETY: the buffer's whole mapping, which nothing else uses.
            unsafe { std::ptr::write_bytes(data, 0, room as usize) };
            return Some((buffer, data, room));
        }
        let (buffer, data) = self.upload_buffer(room)?;
        Some((buffer, data, room))
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        crate::census::made(1, size);
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame"
        );
        late_line(frames);
        card_clock_line(frames, ms);
        crate::shadow::back_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            let staging = STAGING_WAIT_US.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames;
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it), {staging:.1} ms waiting for a staging buffer a submission sent ahead \
                     still copies out of; the rest is reading, hashing and detiling"
                );
            }
            let [through, left] = [0, 1].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT, on by default): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there)"
                );
            }
            let [changes, waiting, under_way] = [0, 1, 2].map(|i| {
                crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64 / [1.0, 1e3, 1e3][i] / frames
            });
            if changes > 0.0 {
                tracing::info!(
                    "changes of title memory's mapping per frame: {changes:.0}, {waiting:.1} ms of them waiting for reads in place, \
                     {under_way:.1} ms under way (each holds {}; VICTORY_MAPPING_RANGES=0 holds all of title memory)",
                    if crate::watch::by_range() { "the bytes it changes" } else { "all of title memory" }
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Of it: looking at slots, spans and refreshes, textures (ms per
        // frame); slots looked at and copies brought up to date per frame.
        let [picked_look, picked_refresh, picked_tex] =
            [0, 1, 2].map(|i| crate::draw::HEAP_PASS_MICROS[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        let [picked_slots, picked_copies] =
            [0, 1].map(|i| crate::draw::HEAP_BUFFER_VISITS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        let picked_mode = if crate::heap_pass::full_pass() { "full" } else { "marked" };
        // Lists whose shaders store through picked buffer views, and the MB
        // of marked views they declared written (`crate::heap_marks`).
        let picked_w = crate::draw::HEAP_WRITE_LISTS.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_w_mb = crate::draw::HEAP_WRITE_BYTES.swap(0, Ordering::Relaxed) as f64 / 1048576.0 / frames;
        let picked_late = crate::draw::HEAP_LATE_REFRESHES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_marked = crate::heap_marks::WRITTEN_SLOTS.load(Ordering::Relaxed);
        let picked_first = crate::heap_marks::FIRST_MARKS.load(Ordering::Relaxed);
        let picked_ro = crate::heap_marks::READ_ONLY_MARKS.load(Ordering::Relaxed);
        // The structure builder's window bytes left in the host copies and
        // declared written (`crate::pipeline::bvh_bound`).
        let [bvh_out, bvh_in] = [0, 1].map(|i| {
            crate::pipeline::BVH_BOUND_BYTES[i].swap(0, Ordering::Relaxed) as f64 / 1048576.0 / frames
        });
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
                let pushed = PUSH_STATS.each_ref().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames);
                if pushed.iter().any(|&v| v > 0.0) {
                    tracing::info!(
                        "what shaders wrote to copies in the card's memory, per frame: {:.0} parts ({:.1} MB) pushed down in their own \
                         submission and merged with no fetch, {:.0} parts fetched because they found no slot (VICTORY_CARD_PUSH=0: \
                         all fetched), {:.0} pushed parts left to a later flight of the same landing that pushed them too, \
                         merged in {:.0} runs of touching parts, {:.0} of those runs changed nothing in title memory; \
                         the runs were of {:.0} copies and took {:.1} ms (VICTORY_WIDE_MERGE=1: copies on several threads)",
                        pushed[0],
                        pushed[1] / 1e6,
                        pushed[2],
                        pushed[3],
                        pushed[4],
                        pushed[5],
                        pushed[6],
                        pushed[7] / 1e3
                    );
                }
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms ({picked_mode}: {picked_slots:.0} slots looked at {picked_look:.1} ms, \
                 {picked_copies:.0} copies refreshed {picked_refresh:.1} ms, textures {picked_tex:.1} ms), {picked_w:.1} lists storing through picked views ({picked_w_mb:.1} MB of marked views declared{picked_off}; {picked_marked} slots marked now, {picked_first} first marks and {picked_ro} marks on read-only views so far), {picked_late:.0} later refreshes; structure builder: {bvh_out:.1} MB of windows kept from title memory, {bvh_in:.1} MB in buffers declared;                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped, looked at again {picked_looks} times ({picked_changed} changed, {picked_changed_late} of them after the scheduled looks); depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_off = if crate::heap_marks::on() { "" } else { " but not copied back: VICTORY_HEAP_WRITES=0" },
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                picked_looks = crate::draw::HEAP_TEXTURES_RECHECKED[0].load(Ordering::Relaxed)
                    + crate::draw::HEAP_TEXTURES_RECHECKED[1].load(Ordering::Relaxed),
                picked_changed = crate::draw::HEAP_TEXTURES_RECHECKED[1].load(Ordering::Relaxed),
                picked_changed_late = crate::draw::HEAP_TEXTURES_RECHECKED[2].load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{}, and for where they lie in their host copy {} shader views off an element, {} unordered ones, {} constant buffers off 256 bytes; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::descriptor::host::MISPLACED_VIEWS[0].load(Ordering::Relaxed),
                    crate::descriptor::host::MISPLACED_VIEWS[1].load(Ordering::Relaxed),
                    crate::descriptor::host::MISPLACED_VIEWS[2].load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            // A texture already a copy source takes no barrier: one from a
            // state to itself is not allowed (upstream #2545).
            let source = state != D3D12_RESOURCE_STATE_COPY_SOURCE;
            let to_dest = transition(
                &back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            if source {
                list.ResourceBarrier(&[
                    transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    to_dest,
                ]);
            } else {
                list.ResourceBarrier(&[to_dest]);
            }
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            let to_present = transition(
                &back,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            if source {
                list.ResourceBarrier(&[
                    transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                    to_present,
                ]);
            } else {
                list.ResourceBarrier(&[to_present]);
            }
            copy.close();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The PCI vendor of the adapter the device is on (0x1002 for AMD), if
    /// known.
    pub(crate) fn vendor_id(&self) -> Option<u32> {
        let adapter = self.adapter.as_ref()?;
        // SAFETY: a plain query of a live adapter.
        unsafe { adapter.GetDesc1() }.ok().map(|d| d.VendorId)
    }

    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        crate::census::made(2, size);
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        crate::census::made(3, size);
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

/// Index of the first list in `runs` that already appeared earlier in
/// `runs` (by identity), or `None` if every list is distinct.
fn first_duplicate(runs: &[&HostList]) -> Option<usize> {
    if runs.len() <= 1 {
        return None;
    }
    let mut seen = std::collections::HashSet::with_capacity(runs.len());
    runs.iter().position(|list| !seen.insert(std::ptr::from_ref(*list) as usize))
}

#[cfg(test)]
mod duplicate_lists {
    /// A submission naming one list twice is split at the second
    /// occurrence and runs to the end (fxceless/victory-fork#675).
    #[test]
    fn a_list_named_twice_is_submitted_in_two_parts() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let a = host.new_list().unwrap();
        a.close();
        let b = host.new_list().unwrap();
        b.close();
        assert_eq!(super::first_duplicate(&[]), None);
        assert_eq!(super::first_duplicate(&[&a]), None);
        assert_eq!(super::first_duplicate(&[&a, &b]), None);
        assert_eq!(super::first_duplicate(&[&a, &a]), Some(1));
        assert_eq!(super::first_duplicate(&[&a, &b, &a]), Some(2));
        host.execute(&[&a, &b, &a]);
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, push_range};
    use std::sync::Arc;

    /// A copy's refresh ranges stay disjoint: what overlaps, or lies within
    /// a page, joins; distant ranges do not bring the span between them;
    /// a range bridging them joins them all (upstream #2428).
    #[test]
    fn refresh_ranges_of_a_copy_stay_disjoint() {
        const P: usize = crate::watch::PAGE;
        let mut rs = vec![(0x1_0000, 0x1_1000)];
        push_range(&mut rs, 0x40_0000, 0x40_1000);
        assert_eq!(rs, [(0x1_0000, 0x1_1000), (0x40_0000, 0x40_1000)]);
        // A page's gap joins ...
        push_range(&mut rs, 0x1_1000 + P, 0x1_2000 + P);
        assert_eq!(rs, [(0x1_0000, 0x1_2000 + P), (0x40_0000, 0x40_1000)]);
        // ... a page and a byte more stays apart, in order.
        push_range(&mut rs, 0x1_2000 + 2 * P + 1, 0x1_2900 + 2 * P + 1);
        assert_eq!(rs.len(), 3);
        assert!(rs.windows(2).all(|w| w[0].1 < w[1].0));
        // Bridging all of them.
        push_range(&mut rs, 0x1_3000 + P, 0x50_0000);
        assert_eq!(rs, [(0x1_0000, 0x50_0000)]);
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn eval_late_met_uses_final_write_and_honors_unsure() {
        use super::eval_late_met;
        use crate::queue::Written;
        let addr = 0x1000usize;
        let eq_one = |v: u64| v == 1;

        // Later overwrite (1 -> 0) invalidates an earlier matching write.
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr, Written::Value(0), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(false)
        );
        // Later matching write (0 -> 1) satisfies the wait without settling.
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(0), 4), (addr, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(true)
        );
        // An overlapping in-flight shader write forces settling (`None`) even
        // when a bottom-of-pipe write matches.
        assert_eq!(
            eval_late_met(
                &[(addr, addr + 4)],
                [(addr, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        // A partial/unaligned overlapping write or timestamp forces settling (`None`).
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr + 2, Written::Value(0), 4)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr, Written::GpuTimestamp, 8)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        // Unrelated writes return `Some(false)` without settling.
        assert_eq!(
            eval_late_met(
                &[(addr + 16, addr + 32)],
                [(addr + 8, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(false)
        );
    }


    #[test]
    fn buffer_states_track_transitions_and_execute_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4};
        use windows::core::Interface;

        // Enable D3D12 debug layer when installed so invalid barriers fail the test.
        let mut debug: Option<ID3D12Debug> = None;
        let _ = unsafe { D3D12GetDebugInterface(&mut debug) };
        if let Some(d) = &debug {
            unsafe { d.EnableDebugLayer() };
        }

        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.unwrap();
        let adapter: IDXGIAdapter = unsafe { factory.EnumWarpAdapter() }.unwrap();
        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.unwrap();
        let device = device.unwrap();
        let info_queue: Option<ID3D12InfoQueue> = device.cast().ok();

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buf,
            )
        }
        .unwrap();
        let buf = buf.unwrap();

        // 1. State-machine transitions and reset back to COMMON.
        let mut states = super::BufferStates::default();
        assert_eq!(
            states.transition_for(&buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            Some(D3D12_RESOURCE_STATE_COMMON)
        );
        assert_eq!(states.transition_for(&buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), None);
        assert_eq!(
            states.transition_for(&buf, D3D12_RESOURCE_STATE_GENERIC_READ),
            Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        );
        assert_eq!(states.transition_for(&buf, D3D12_RESOURCE_STATE_GENERIC_READ), None);
        assert_eq!(
            states.transition_for(&buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            Some(D3D12_RESOURCE_STATE_GENERIC_READ)
        );
        states.clear();
        assert_eq!(
            states.transition_for(&buf, D3D12_RESOURCE_STATE_GENERIC_READ),
            Some(D3D12_RESOURCE_STATE_COMMON)
        );

        // 2. Real D3D12 command list: UAV -> GENERIC_READ -> VB/IB -> UAV across two submissions.
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }.unwrap();
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();
        let cmd: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
        }
        .unwrap();
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let gpu_va = unsafe { buf.GetGPUVirtualAddress() };

        for pass in 1u64..=2 {
            let mut bs = super::BufferStates::default();
            for wanted in [
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            ] {
                if let Some(before) = bs.transition_for(&buf, wanted) {
                    let barrier = super::transition(&buf, before, wanted);
                    unsafe { cmd.ResourceBarrier(&[barrier]) };
                }
            }
            let vbv = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu_va,
                SizeInBytes: 128,
                StrideInBytes: 16,
            };
            let ibv = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: gpu_va + 128,
                SizeInBytes: 128,
                Format: DXGI_FORMAT_R32_UINT,
            };
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[vbv]));
                cmd.IASetIndexBuffer(Some(&ibv));
            }
            if let Some(before) = bs.transition_for(&buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                let barrier = super::transition(&buf, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                unsafe { cmd.ResourceBarrier(&[barrier]) };
            }
            unsafe { cmd.Close() }.unwrap();
            let list_ref: ID3D12CommandList = cmd.cast().unwrap();
            unsafe { queue.ExecuteCommandLists(&[Some(list_ref)]) };
            unsafe { queue.Signal(&fence, pass) }.unwrap();
            while unsafe { fence.GetCompletedValue() } < pass {
                std::thread::yield_now();
            }
            if pass == 1 {
                unsafe {
                    allocator.Reset().unwrap();
                    cmd.Reset(&allocator, None).unwrap();
                }
            }
        }
        assert_eq!(unsafe { device.GetDeviceRemovedReason() }, Ok(()));
        if let Some(q) = info_queue {
            let n = unsafe { q.GetNumStoredMessages() };
            let mut errors = Vec::new();
            for i in 0..n {
                let mut len = 0usize;
                let _ = unsafe { q.GetMessage(i, None, &mut len) };
                if len == 0 {
                    continue;
                }
                let mut raw = vec![0u8; len];
                let msg = raw.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if unsafe { q.GetMessage(i, Some(msg), &mut len) }.is_ok() {
                    let m = unsafe { &*msg };
                    if m.Severity == D3D12_MESSAGE_SEVERITY_ERROR
                        || m.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                    {
                        let text = unsafe {
                            std::slice::from_raw_parts(
                                m.pDescription,
                                m.DescriptionByteLength.saturating_sub(1),
                            )
                        };
                        errors.push(String::from_utf8_lossy(text).into_owned());
                    }
                }
            }
            assert!(errors.is_empty(), "D3D12 debug errors: {errors:?}");
        }
    }
}

#[cfg(test)]
mod fill_on_card_tests {
    use super::*;

    /// A readback buffer of `len` bytes.
    fn readback(host: &Host, len: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
                .unwrap();
        }
        buffer.unwrap()
    }

    /// What a flight's shaders wrote to a host copy (bytes the title's memory
    /// does not have) reaches constants filled on the card, byte for byte,
    /// for two fills in one list.
    #[test]
    fn constants_filled_on_the_card_read_what_the_card_wrote() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        const LEN: usize = 1 << 16;
        // Title memory: a buffer the title fills with 0xAA.
        let title: &'static mut [u8] = Box::leak(vec![0xAAu8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0001usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start + 64, 512).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy starts at the buffer");
        // A "flight" writes a pattern into the copy on the card.
        let pattern: Vec<u8> = (0..1024u32).map(|i| (i * 7 + 3) as u8).collect();
        let (upload, data) = host.upload_buffer(1024).unwrap();
        // SAFETY: the upload buffer holds 1024 mapped bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024) };
        let flight = host.new_list().unwrap();
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe { flight.list.CopyBufferRegion(&shadow.buffer, offset + 64, &upload, 0, 1024) };
        flight.close();
        host.execute(&[&flight]);
        // The next list fills two constant buffers from it on the card.
        let list = host.new_list().unwrap();
        let first = list.card_constants(&shadow, start + 64, 512).expect("room");
        let second = list.card_constants(&shadow, start + 64 + 512, 300).expect("room");
        assert_eq!(first % 256, 0);
        assert_eq!(second % 256, 0);
        let constants = list.card_constants.lock().unwrap().as_ref().unwrap().0.clone();
        // SAFETY: a live buffer.
        let base = unsafe { constants.GetGPUVirtualAddress() };
        let back = readback(host, 1024);
        // SAFETY: an open list; the constants buffer is in the constant
        // buffer state after the fills; ranges inside both buffers.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &constants,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&back, 0, &constants, first - base, 512);
            list.list.CopyBufferRegion(&back, 512, &constants, second - base, 300);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 1024 bytes.
        let read = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024).to_vec()
        };
        assert_eq!(read[..512], pattern[..512], "the first fill");
        assert_eq!(read[512..812], pattern[512..812], "the second fill");
        // The title's memory was never read: it still holds 0xAA.
        assert!(title[64..1088].iter().all(|&b| b == 0xAA));
        crate::shadow::unregister(resource, start);
    }

    /// A copy out of a range this list's shaders write is staged from the
    /// host copy on the card, after what the list wrote there before (a
    /// store into the copy stands in for the shader), not from title memory
    /// as it was before the list (PalmVIDev 4d1ce258); a range no shader of
    /// the list writes is not staged.
    #[test]
    fn a_copy_source_the_list_wrote_is_staged_from_the_host_copy() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        const LEN: usize = 1 << 16;
        let title: &'static mut [u8] = Box::leak(vec![0x11u8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0002usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start, LEN as u64).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy starts at the buffer");
        let list = host.new_list().unwrap();
        // Nothing declared written: no staging.
        assert!(list.staged_written_source(host, start + 256, 256).is_none());
        // The "shader" declares and writes 256 bytes of the copy.
        list.defer_gpu_write(&shadow, start as usize + 256, 256);
        let pattern: Vec<u8> = (0..256u32).map(|i| (i * 5 + 1) as u8).collect();
        let (upload, data) = host.upload_buffer(256).unwrap();
        // SAFETY: the upload buffer holds 256 mapped bytes; an open list,
        // live buffers, ranges inside them.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 256);
            list.list.CopyBufferRegion(&shadow.buffer, offset + 256, &upload, 0, 256);
        }
        let staged = list.staged_written_source(host, start + 256, 256).expect("staged");
        let back = readback(host, 256);
        // SAFETY: an open list; the staged buffer is in the copy source state.
        unsafe { list.list.CopyBufferRegion(&back, 0, &staged, 0, 256) };
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 256 bytes.
        let read = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 256).to_vec()
        };
        assert_eq!(read, pattern, "what the list wrote, not the title's 0x11");
        crate::shadow::unregister(resource, start);
    }
}

#[cfg(test)]
mod fills {
    use super::{HostList, host};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    fn list() -> Option<HostList> {
        host()?.new_list().ok()
    }

    /// Fills and copies run in recorded order, and a fill's unmapped pages
    /// read as zeros while its mapped ones are copied.
    #[test]
    fn steps_run_in_order_and_unmapped_pages_read_as_zeros() {
        let Some(list) = list() else {
            return; // no host GPU
        };
        // Three reserved pages, the middle one not committed.
        // SAFETY: a fresh reservation; the committed pages are ours to fill.
        let holed = unsafe {
            let at =
                VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!at.is_null());
            for page in [0, 2] {
                let page = at.add(page * PAGE);
                assert!(!VirtualAlloc(page.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                page.write_bytes(7, PAGE);
            }
            at
        };
        let old = [1u8; 256];
        let new = [2u8; 256];
        let mut title = old;
        let (mut first, mut second) = ([0u8; 256], [0u8; 100]);
        let mut whole = vec![9u8; 3 * PAGE];
        let memory = title.as_mut_ptr();
        list.defer_fill(first.as_mut_ptr(), memory, 256);
        list.defer_fill(whole.as_mut_ptr(), holed, 3 * PAGE);
        // The title's own copy: fills after it see its bytes.
        list.defer_copy(memory, new.as_ptr(), 256);
        list.defer_fill(second.as_mut_ptr(), memory, 100);
        list.run_deferred(&mut super::Batch::default());
        assert_eq!((first, second), (old, [2u8; 100]));
        assert_eq!(title, new);
        assert!(whole[..PAGE].iter().all(|&b| b == 7));
        assert!(whole[PAGE..2 * PAGE].iter().all(|&b| b == 0));
        assert!(whole[2 * PAGE..].iter().all(|&b| b == 7));
        // SAFETY: the reservation above.
        assert_ne!(unsafe { VirtualFree(holed.cast(), 0, MEM_RELEASE) }, 0);
    }

    /// A list's constant buffer fills, as a submission runs them. Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture constant_fills`.
    #[test]
    #[ignore = "benchmark"]
    fn constant_fills_per_list() {
        let Some(list) = list() else {
            return; // no host GPU
        };
        const FILLS: usize = 4096;
        let title: Vec<u8> = (0..FILLS * 512).map(|i| (i * 31 + i / 512) as u8).collect();
        for len in [64usize, 256, 512] {
            list.close();
            list.reset();
            let mut filled = Vec::new();
            for i in 0..FILLS {
                let (_, to) = list.scratch(len as u64).expect("scratch");
                list.defer_fill(to, title[i * 512..].as_ptr(), len);
                filled.push((to, i * 512));
            }
            let mut best = std::time::Duration::MAX;
            for _ in 0..50 {
                let started = std::time::Instant::now();
                list.run_deferred(&mut super::Batch::default());
                best = best.min(started.elapsed());
            }
            for (to, at) in filled {
                // SAFETY: the list's scratch memory.
                let got = unsafe { std::slice::from_raw_parts(to, len) };
                assert_eq!(got, &title[at..at + len]);
            }
            let each = best.as_nanos() as f64 / FILLS as f64;
            println!("{FILLS} fills of {len} B: {best:?} ({each:.1} ns each)");
        }
    }
}

