use std::cell::Cell;
use std::sync::mpsc::{Sender, channel};
use std::time::Duration;

use windows::Win32::Foundation::E_FAIL;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4};
use windows::core::{Error, Result, s};

use super::*;

#[derive(Debug, PartialEq, Eq)]
enum Peer {
    WaitingForName,
    Creating,
}

thread_local! {
    static ON_NAME_WAIT: Cell<Option<Sender<Peer>>> = const { Cell::new(None) };
}

// Signal from the contended path, so a timeout is never evidence of waiting.
pub(super) fn waiting_for_name() {
    ON_NAME_WAIT.with(|signal| {
        if let Some(signal) = signal.take() {
            signal.send(Peer::WaitingForName).unwrap();
        }
    });
}

struct Compute {
    name: String,
    code: Vec<u8>,
}

impl Compute {
    fn new(root_blob: &[u8], threads: u32) -> Self {
        let source = format!("[numthreads({threads}, 1, 1)] void main() {{}}");
        let code = crate::host::compile(&source, s!("main"), s!("cs_5_1")).unwrap();
        Self {
            name: name("compute", root_blob, &[&code], ""),
            code,
        }
    }

    fn request(
        &self,
        host: &crate::host::Host,
        root: &ID3D12RootSignature,
        before_create: impl FnOnce() -> Result<()>,
    ) -> Result<bool> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer, as in the host pipeline descriptor.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: self.code.as_ptr().cast(),
                BytecodeLength: self.code.len(),
            },
            ..Default::default()
        };
        let mut created = false;
        let mut before_create = Some(before_create);
        pipeline(
            &self.name,
            // SAFETY: the descriptor and its bytecode stay live through the calls.
            |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
            || {
                created = true;
                before_create.take().map_or(Ok(()), |f| f())?;
                // SAFETY: as above.
                unsafe { host.device.CreateComputePipelineState(&desc) }
            },
        )?;
        Ok(created)
    }
}

fn overlap(
    host: &crate::host::Host,
    root: &ID3D12RootSignature,
    first: &Compute,
    second: &Compute,
) -> (Peer, bool, bool) {
    const WATCHDOG: Duration = Duration::from_secs(30);
    std::thread::scope(|scope| {
        let (entered, ready) = channel();
        let (release, continue_create) = channel();
        let first = scope.spawn(move || {
            first.request(host, root, || {
                entered.send(()).unwrap();
                continue_create.recv().unwrap();
                Ok(())
            })
        });
        ready.recv_timeout(WATCHDOG).expect("first request must miss and enter create");

        let (send_progress, progress) = channel();
        let second = scope.spawn(move || {
            ON_NAME_WAIT.set(Some(send_progress.clone()));
            second.request(host, root, || {
                send_progress.send(Peer::Creating).unwrap();
                Ok(())
            })
        });
        // Both outcomes release A. B either encounters A's held name or has
        // already missed the library and entered its own create closure.
        let event = progress.recv_timeout(WATCHDOG);
        release.send(()).unwrap();
        let first_created = first.join().unwrap().unwrap();
        let second_created = second.join().unwrap().unwrap();
        (
            event.expect("second request must report progress"),
            first_created,
            second_created,
        )
    })
}

// Run on its own with a D3D12 adapter that supports compute pipeline libraries.
#[test]
#[ignore = "requires a D3D12 compute pipeline library"]
fn concurrent_library_misses_are_serialized_by_name() {
    let host = crate::host::host().expect("D3D12 host");
    let library = library().expect("D3D12 pipeline library");
    // SAFETY: plain DXGI queries against the selected D3D12 device.
    let adapter = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
            .and_then(|adapter| adapter.GetDesc1())
            .unwrap()
    };
    let adapter_name = String::from_utf16_lossy(&adapter.Description);
    println!(
        "DXGI {} {:04X}:{:04X}",
        adapter_name.trim_end_matches('\0'),
        adapter.VendorId,
        adapter.DeviceId,
    );

    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: a valid empty root description and a live blob output.
    let (root, root_blob) = unsafe {
        D3D12SerializeRootSignature(
            &D3D12_ROOT_SIGNATURE_DESC::default(),
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            None,
        )
        .unwrap();
        let blob = blob.unwrap();
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        (host.device.CreateRootSignature::<ID3D12RootSignature>(0, bytes).unwrap(), bytes.to_vec())
    };

    let same = Compute::new(&root_blob, 1);
    let before = library.added.load(Ordering::Relaxed);
    let same_result = overlap(host, &root, &same, &same);
    let same_stores = library.added.load(Ordering::Relaxed) - before;
    println!("same name: {same_result:?}, successful stores={same_stores}");

    // Use different host names that would share the old striped load lock.
    let old_stripe = |name: &str| {
        name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b))) % 64
    };
    let first = Compute::new(&root_blob, 2);
    let second = (3..1024)
        .map(|threads| Compute::new(&root_blob, threads))
        .find(|other| other.name != first.name && old_stripe(&other.name) == old_stripe(&first.name))
        .expect("different name with the same old stripe");
    let before = library.added.load(Ordering::Relaxed);
    let distinct_result = overlap(host, &root, &first, &second);
    let distinct_stores = library.added.load(Ordering::Relaxed) - before;
    println!("distinct names on the same old stripe: {distinct_result:?}, successful stores={distinct_stores}");

    let retry = Compute::new(&root_blob, 1024);
    let failed = retry.request(host, &root, || Err(Error::from_hresult(E_FAIL)));
    assert_eq!(failed.unwrap_err().code(), E_FAIL);
    assert!(retry.request(host, &root, || Ok(())).unwrap());
    assert!(!retry.request(host, &root, || Ok(())).unwrap());
    println!("failed create: retry created and a later request loaded");

    assert_eq!(distinct_result, (Peer::Creating, true, true));
    assert_eq!(distinct_stores, 2);
    assert_eq!(same_stores, 1);
    assert_eq!(same_result, (Peer::WaitingForName, true, false));
}
