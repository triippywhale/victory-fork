//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Whether the depth bounds test is on ([`bounds_pipeline`]).
    bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::target_format(crate::resource::pc_format(
                            f.0 as u32,
                        )) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Makes the whole state one D3D12 accepts where the console is more
    /// lenient; called once every subobject of the stream is read, before
    /// the state is described or used.
    ///
    /// Blending asked for on an integer target is turned off for that
    /// target: the console leaves such a target unblended, while D3D12
    /// refuses the whole pipeline (E_INVALIDARG; a mesh pipeline with
    /// R32_FLOAT, B4G4R4A4_UNORM and R8_UINT targets, all three blending,
    /// was refused by an AMD host, and its draws were dropped).
    pub(crate) fn settle(&mut self) {
        let count = (self.targets.NumRenderTargets as usize).min(self.targets.RTFormats.len());
        let independent = self.blend.IndependentBlendEnable.as_bool();
        let integers: Vec<usize> = (0..count)
            .filter(|&i| {
                let state = &self.blend.RenderTarget[if independent { i } else { 0 }];
                integer(self.targets.RTFormats[i]) && state.BlendEnable.as_bool()
            })
            .collect();
        if integers.is_empty() {
            return;
        }
        // One state for every target becomes one each, so that the targets
        // that can blend still do.
        if !independent {
            self.blend.RenderTarget = [self.blend.RenderTarget[0]; 8];
            self.blend.IndependentBlendEnable = true.into();
        }
        for i in integers {
            self.blend.RenderTarget[i].BlendEnable = false.into();
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate
                )
            })
            .collect();
        // (Pipelines without the test keep the names they had.)
        let bounds = if self.bounds { " bounds" } else { "" };
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}{bounds}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether the title's root SRVs are bound as one-view tables of the
/// buffer's host copy, whose reads past its end return zeros, rather than
/// as root descriptors, which have no size: a mesh shader's vertex fetch
/// has no record limit, and a read past the copy reaches whatever the host
/// placed after it (memory it has freed: the device is removed, upstream
/// PR #493). `VICTORY_ROOT_SRV_DIRECT=1` goes back to root descriptors.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root UAVs: one-view tables like the SRVs (or root descriptors with
    /// `VICTORY_ROOT_SRV_DIRECT=1`).
    pub(crate) root_uavs: Vec<(u8, u32)>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Has the geometry stage that completes the console's rectangles
    /// (`victory_shader::emit::rect_list_gs`): its rectangle draws go out
    /// as the title gives them, three vertices a rectangle.
    pub(crate) rects: bool,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Its depth, stencil, cull and blend state in short, for the frame
    /// listing (empty for compute).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) state: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        // For the report when the host device is removed: what this thread
        // has in the library from here on.
        super::library::label(format!("{shaders:?}"));
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)", shaders);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // The ranges table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    let uav = kind == D3D12_ROOT_PARAMETER_TYPE_UAV;
                    // Ours, not the title's, stay root descriptors: the
                    // table of strides, the directory of title memory by
                    // address and its fault notes, the marks of heap slots
                    // stored through.
                    let ours = if uav {
                        slot == victory_shader::emit::SLOT_FAULTS
                            || slot == victory_shader::emit::SLOT_HEAP_MARKS
                    } else {
                        slot == victory_shader::emit::SLOT_STRIDES
                            || slot == victory_shader::emit::SLOT_MEMORY
                    };
                    if (uav || kind == D3D12_ROOT_PARAMETER_TYPE_SRV) && !ours && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the
                        // reads (and a root UAV's writes), which a root
                        // descriptor leaves unbounded.
                        // (Our own table of strides stays a descriptor.)
                        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                            RangeType: if uav {
                                D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                            } else {
                                D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                            },
                            NumDescriptors: 1,
                            BaseShaderRegister: u32::from(slot),
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                            OffsetInDescriptorsFromTableStart: 0,
                        }));
                        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last()?;
                        out.push((slot, params.len() as u32));
                        params.push(D3D12_ROOT_PARAMETER1 {
                            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                                    NumDescriptorRanges: 1,
                                    pDescriptorRanges: range,
                                },
                            },
                            ShaderVisibility: visibility,
                        });
                        continue;
                    }
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, compute.is_some()) {
            Ok(root) => root,
            Err(e) => {
                // Its size in dwords (64 at most): constants one each, root
                // views two, tables one.
                let dwords: u32 = params
                    .iter()
                    .map(|p| match p.ParameterType {
                        // SAFETY: the union member the type names.
                        D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => unsafe { p.Anonymous.Constants.Num32BitValues },
                        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE => 1,
                        _ => 2,
                    })
                    .sum();
                tracing::warn!(error = %e, dwords, ?shaders, "host root signature refused");
                return None;
            }
        };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        let none = D3D12_SHADER_BYTECODE::default;
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if let Some(ms) = mesh {
            // A mesh shader whose outputs pass the 32 KiB D3D12 allows:
            // it and the pixel shader translated together, its 16-bit
            // outputs packed (#283), when both translate so and the host's
            // pixel shaders read barycentrics. Otherwise the stages as
            // translated alone (drivers that take outputs past the limit
            // draw them).
            let linked = ms
                .mesh_link()
                .filter(|_| host.barycentrics())
                .and_then(|link| {
                    let ms = ms.linked_dxil(link)?;
                    let ps = match stages.get(1) {
                        Some((ps, _)) => Some(ps.linked_dxil(link)?),
                        None => None,
                    };
                    Some((ms, ps))
                });
            let bytecode = |dxil: &[u8]| D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            };
            let (ms_code, ps_code) = match &linked {
                Some((ms, ps)) => (
                    bytecode(ms),
                    ps.as_deref()
                        .map_or(D3D12_SHADER_BYTECODE::default(), |ps| bytecode(ps)),
                ),
                None => (code(0), code(1)),
            };
            // "triangles": the stream names its topology since it did not
            // load from the library without one (see `mesh_pipeline`); a
            // new name keeps such entries, stored before, from being
            // looked up again.
            let named = |ms: &D3D12_SHADER_BYTECODE, ps: &D3D12_SHADER_BYTECODE| {
                super::library::name(
                    "mesh triangles",
                    &root_blob,
                    &[&code_bytes(ms), &code_bytes(ps)],
                    &state,
                )
            };
            // With the depth bounds test on, the stream form (`bounds_pipeline`).
            let make = |ms: D3D12_SHADER_BYTECODE, ps: D3D12_SHADER_BYTECODE| {
                if g.bounds && depth_bounds_test() {
                    bounds_pipeline(&host.device, &root, [none(), none(), ms, ps], g, &named(&ms, &ps))
                } else {
                    mesh_pipeline(&host.device, &root, ms, ps, g, Some(&named(&ms, &ps)))
                }
            };
            let made = match make(ms_code, ps_code) {
                // A linked pair the host refuses (a pixel shader that
                // declares a packed semantic otherwise): the stages as
                // translated alone.
                Err(e) if linked.is_some() => {
                    tracing::warn!(error = %e, ?shaders, "linked host mesh pipeline refused; trying the stages as translated alone");
                    host.log_messages();
                    make(code(0), code(1))
                }
                made => made,
            };
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    // Without the debug layer the driver gives no reason:
                    // which part of the description it refuses is found by
                    // trying again with parts of it made neutral.
                    let refused = mesh_refusal(&host.device, &root, code(0), code(1), g);
                    tracing::warn!(
                        error = %e,
                        ?shaders,
                        root_parameters = params.len(),
                        static_samplers = samplers.len(),
                        refused,
                        state,
                        "host mesh pipeline refused"
                    );
                    // The debug layer's messages, if any, of every attempt.
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
            });
        }

        // A pipeline the title makes for rectangles gets a geometry stage
        // that completes each one from the three vertices the vertex
        // shader made. Without it (refused here or by the host) the draws
        // fall back on what `draw` does for rectangles.
        let mut rect_gs = None;
        if g.topology.0 == RECTANGLES {
            let ps = stages.get(1).and_then(|(s, _)| s.translated());
            let vs = &stages[0].0.translated()?.host.dxil;
            match victory_shader::emit::rect_list_gs(vs, ps.map(|t| &t.host.dxil[..])) {
                Ok(gs) => rect_gs = Some(gs),
                Err(e) => tracing::warn!(error = %e, ?shaders, "no rectangle geometry stage"),
            }
        }
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: rect_gs.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |gs| {
                D3D12_SHADER_BYTECODE {
                    pShaderBytecode: gs.as_ptr().cast(),
                    BytecodeLength: gs.len(),
                }
            }),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        zero_padding(&mut desc.BlendState, &mut desc.DepthStencilState);
        // A pipeline with the geometry stage has a name of its own in the
        // library; the others keep theirs.
        let name = |gs: Option<&Vec<u8>>| match gs {
            Some(gs) => super::library::name(
                "graphics rectangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs[..]],
                &state,
            ),
            None => super::library::name(
                "graphics",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            ),
        };
        let make = |desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, name: &str| {
            if g.bounds && depth_bounds_test() {
                return bounds_pipeline(&host.device, &root, [desc.VS, desc.GS, none(), desc.PS], g, name);
            }
            super::library::pipeline(
                name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(desc) },
            )
        };
        let mut made = make(&desc, &name(rect_gs.as_ref()));
        if made.is_err() && rect_gs.is_some() {
            // The host refused the geometry stage: once more without it.
            tracing::warn!(?shaders, "host pipeline refused with the rectangle geometry stage");
            host.log_messages();
            desc.GS = D3D12_SHADER_BYTECODE::default();
            rect_gs = None;
            made = make(&desc, &name(None));
        }
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        #[cfg(feature = "diagnostics")]
        tracing::info!(?shaders, state = %g.describe(), "host pipeline");
        #[cfg(not(feature = "diagnostics"))]
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            rects: rect_gs.is_some(),
            state: {
                let (d, r, b) = (&g.depth, &g.rasterizer, &g.blend.RenderTarget[0]);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC| {
                    format!(
                        "fail {} depth-fail {} pass {} func {}",
                        f.StencilFailOp.0, f.StencilDepthFailOp.0, f.StencilPassOp.0, f.StencilFunc.0
                    )
                };
                format!(
                    "depth {} write {} func {}; stencil {} read {:#x} write {:#x} front [{}] back [{}]; cull {} front-ccw {}; blend {} src {} dst {} op {} mask {:#x}",
                    d.DepthEnable.as_bool(),
                    d.DepthWriteMask.0,
                    d.DepthFunc.0,
                    d.StencilEnable.as_bool(),
                    d.StencilReadMask,
                    d.StencilWriteMask,
                    face(&d.FrontFace),
                    face(&d.BackFace),
                    r.CullMode.0,
                    r.FrontCounterClockwise.as_bool(),
                    b.BlendEnable.as_bool(),
                    b.SrcBlend.0,
                    b.DestBlend.0,
                    b.BlendOp.0,
                    b.RenderTargetWriteMask
                )
            },
        })
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout).
///
/// The stream names a triangle topology, which D3D12 does not need for a
/// mesh shader: without it the pipeline library refuses to load a stored
/// pipeline with the very stream it was made from (`E_INVALIDARG`; debug
/// layer message 972, "does not match the one used to create the PSO"), so
/// every mesh pipeline was compiled again, and its store then refused (970,
/// "already exists"). The translation makes triangle mesh shaders only
/// (`mesh_prologue` refuses others), and the debug layer wants the two to
/// agree (1323).
///
/// With a `name`, the pipeline is loaded from the library or stored there;
/// without one it is only made.
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: Option<&str>,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
    };
    zero_padding(&mut stream.blend.1, &mut stream.depth.1);
    // The gaps between a subobject's type and its description, and after it.
    fn zero_gaps<T>(sub: &mut Sub<T>) {
        let at = std::mem::offset_of!(Sub<T>, 1);
        let bytes = (&raw mut *sub).cast::<u8>();
        // SAFETY: padding bytes of `sub`, inside it.
        unsafe {
            bytes.add(4).write_bytes(0, at - 4);
            bytes.add(at + size_of::<T>()).write_bytes(0, size_of::<Sub<T>>() - at - size_of::<T>());
        }
    }
    zero_gaps(&mut stream.root);
    zero_gaps(&mut stream.ms);
    zero_gaps(&mut stream.ps);
    zero_gaps(&mut stream.blend);
    zero_gaps(&mut stream.mask);
    zero_gaps(&mut stream.raster);
    zero_gaps(&mut stream.depth);
    zero_gaps(&mut stream.targets);
    zero_gaps(&mut stream.dsv);
    zero_gaps(&mut stream.samples);
    zero_gaps(&mut stream.topology);
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    // SAFETY: the stream holds whole subobjects pointing at live data.
    let create = || unsafe { device.CreatePipelineState(&desc) };
    match name {
        Some(name) => super::library::pipeline(
            name,
            // SAFETY: as above.
            |library, name| unsafe { library.LoadPipeline(name, &desc) },
            create,
        ),
        None => create(),
    }
}

/// Which part of a refused mesh pipeline's description the driver refuses:
/// it is made again (outside the library) with one part after another made
/// neutral, and the part whose neutral form is first accepted is named.
/// Each step keeps the ones before it, so the part named is one the
/// pipeline needs changed once those before it are neutral too.
fn mesh_refusal(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
) -> &'static str {
    let neutral = GraphicsState::default();
    let mut s = GraphicsState {
        blend: g.blend,
        sample_mask: g.sample_mask,
        rasterizer: g.rasterizer,
        depth: g.depth,
        targets: g.targets,
        depth_format: g.depth_format,
        samples: g.samples,
        ..GraphicsState::default()
    };
    let accepted = |s: &GraphicsState, ps| mesh_pipeline(device, root, ms, ps, s, None).is_ok();
    s.blend = neutral.blend;
    s.sample_mask = neutral.sample_mask;
    if accepted(&s, ps) {
        return "the blend state or sample mask";
    }
    s.rasterizer = neutral.rasterizer;
    if accepted(&s, ps) {
        return "the rasterizer state";
    }
    // Off rather than D3D12's default, which tests depth: valid with any
    // depth format, and with none (the next step).
    s.depth = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: false.into(),
        ..neutral.depth
    };
    if accepted(&s, ps) {
        return "the depth and stencil state";
    }
    s.targets = D3D12_RT_FORMAT_ARRAY::default();
    s.targets.NumRenderTargets = 1;
    s.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    s.depth_format = DXGI_FORMAT_UNKNOWN;
    s.samples = neutral.samples;
    if accepted(&s, ps) {
        return "the target formats, depth format or sample count";
    }
    // A mesh pipeline may leave the pixel stage out.
    if accepted(&s, D3D12_SHADER_BYTECODE::default()) {
        return "the pixel shader or how it links to the mesh shader";
    }
    "the mesh shader or the root signature"
}

/// Whether the host has the depth bounds test (said once if not).
/// `VICTORY_DEPTH_BOUNDS=0` goes without it, as before.
pub(crate) fn depth_bounds_test() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        if std::env::var_os("VICTORY_DEPTH_BOUNDS").is_some_and(|v| v == "0") {
            return false;
        }
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        // SAFETY: the description matches the feature asked about.
        let has = host().is_some_and(|host| unsafe {
            host.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS2,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok()
        }) && options.DepthBoundsTestSupported.as_bool();
        if !has {
            tracing::warn!("no depth bounds test on this host: draws that ask for it pass at every depth");
        }
        has
    })
}

/// A graphics pipeline (vertex and geometry, or mesh, then pixel code) with
/// the depth bounds test, which only the stream form can ask for.
fn bounds_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    [vs, gs, ms, ps]: [D3D12_SHADER_BYTECODE; 4],
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    /// Appends one to the stream, which is whole words.
    fn add<T>(stream: &mut Vec<u64>, kind: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, value: T) {
        let words = size_of::<Sub<T>>() / 8;
        let at = stream.len();
        stream.resize(at + words, 0);
        // SAFETY: room for one subobject, whose alignment is the words';
        // the descriptions are plain data (nothing is dropped).
        unsafe { stream.as_mut_ptr().add(at).cast::<Sub<T>>().write(Sub(kind, value)) };
    }
    let mut stream = Vec::new();
    let s = &mut stream;
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw());
    for (kind, code) in [
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs),
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs),
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
    ] {
        if !code.pShaderBytecode.is_null() {
            add(s, kind, code);
        }
    }
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend);
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask);
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer);
    let d = &g.depth;
    let depth = D3D12_DEPTH_STENCIL_DESC1 {
        DepthEnable: d.DepthEnable,
        DepthWriteMask: d.DepthWriteMask,
        DepthFunc: d.DepthFunc,
        StencilEnable: d.StencilEnable,
        StencilReadMask: d.StencilReadMask,
        StencilWriteMask: d.StencilWriteMask,
        FrontFace: d.FrontFace,
        BackFace: d.BackFace,
        DepthBoundsTestEnable: true.into(),
    };
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth);
    if ms.pShaderBytecode.is_null() {
        let inputs = D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: g.inputs.as_ptr(),
            NumElements: g.inputs.len() as u32,
        };
        add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, inputs);
        add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, g.strip_cut);
    }
    let topology = match g.topology.0 {
        _ if !ms.pShaderBytecode.is_null() => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => g.topology,
    };
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, topology);
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets);
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format);
    add(s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples);
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: stream.len() * 8,
        pPipelineStateSubobjectStream: stream.as_mut_ptr().cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// Whether a target format holds integers, which D3D12 does not blend.
fn integer(format: DXGI_FORMAT) -> bool {
    matches!(
        format,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// Zeroes the padding in a blend and a depth-stencil description: the
/// title's (copied whole by [`GraphicsState::read`]) holds whatever its
/// memory held, and the library compares a loaded pipeline's description
/// with the stored one byte for byte, so a pipeline whose padding changed
/// between runs was refused, made again and its store refused too, every
/// run (upstream Marvisak/VIctory#1690). Done on the description handed to
/// the host, after which nothing moves it.
fn zero_padding(blend: &mut D3D12_BLEND_DESC, depth: &mut D3D12_DEPTH_STENCIL_DESC) {
    use std::mem::offset_of;
    let mask = offset_of!(D3D12_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask) + 1;
    for target in &mut blend.RenderTarget {
        let bytes = (&raw mut *target).cast::<u8>();
        // SAFETY: the padding after the last field, inside `target`.
        unsafe { bytes.add(mask).write_bytes(0, size_of::<D3D12_RENDER_TARGET_BLEND_DESC>() - mask) };
    }
    let (write, front) = (
        offset_of!(D3D12_DEPTH_STENCIL_DESC, StencilWriteMask) + 1,
        offset_of!(D3D12_DEPTH_STENCIL_DESC, FrontFace),
    );
    // SAFETY: the padding between two fields, inside `depth`.
    unsafe { (&raw mut *depth).cast::<u8>().add(write).write_bytes(0, front - write) };
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }.to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let heap_indexing = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    let desc_with = |flags| D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: flags,
            },
        },
    };
    let desc = desc_with(input | heap_indexing);
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(ref error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!(
                    "root signature: runtime rejected directly-indexed flags ({}), retrying without them",
                    String::from_utf8_lossy(text)
                );
            }
            drop(error.take());
            blob = None;
            let retry = desc_with(input);
            let mut retry_error: Option<ID3DBlob> = None;
            if let Err(e2) =
                D3D12SerializeVersionedRootSignature(&retry, &mut blob, Some(&mut retry_error))
            {
                if let Some(error) = retry_error {
                    let text = std::slice::from_raw_parts(
                        error.GetBufferPointer().cast::<u8>(),
                        error.GetBufferSize(),
                    );
                    tracing::warn!("root signature retry: {}", String::from_utf8_lossy(text));
                }
                return Err(e2);
            }
            let _ = e;
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root = match device.CreateRootSignature(0, bytes) {
            Ok(root) => (root, bytes.to_vec()),
            Err(e) => {
                let retry = desc_with(input);
                let mut fallback_blob: Option<ID3DBlob> = None;
                if D3D12SerializeVersionedRootSignature(&retry, &mut fallback_blob, None).is_ok()
                    && let Some(fallback_blob) = fallback_blob
                {
                    let fallback_bytes = std::slice::from_raw_parts(
                        fallback_blob.GetBufferPointer().cast::<u8>(),
                        fallback_blob.GetBufferSize(),
                    );
                    let root = device.CreateRootSignature(0, fallback_bytes)?;
                    (root, fallback_bytes.to_vec())
                } else {
                    return Err(e);
                }
            }
        };
        Ok(root)
    }
}

#[cfg(test)]
mod root_flag_tests {
    use super::*;

    #[test]
    fn host_root_signature_serializes_and_creates() {
        let Some(host) = crate::host::host() else { return };
        let (root, bytes) = root_signature(&host.device, &[], &[], false).unwrap();
        assert!(!bytes.is_empty());
        let _ = root;
        let (comp_root, comp_bytes) = root_signature(&host.device, &[], &[], true).unwrap();
        assert!(!comp_bytes.is_empty());
        let _ = comp_root;
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn a_description_s_padding_is_zero_whatever_the_title_left_in_it() {
        fn bytes<T>(v: &T) -> &[u8] {
            // SAFETY: the bytes of a plain description.
            unsafe { std::slice::from_raw_parts((&raw const *v).cast::<u8>(), size_of::<T>()) }
        }
        let states = |fill: u8| {
            // SAFETY: plain data; every byte, padding too, is `fill` first.
            let (mut blend, mut depth): (D3D12_BLEND_DESC, D3D12_DEPTH_STENCIL_DESC) =
                unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
            unsafe {
                (&raw mut blend).cast::<u8>().write_bytes(fill, size_of::<D3D12_BLEND_DESC>());
                (&raw mut depth).cast::<u8>().write_bytes(fill, size_of::<D3D12_DEPTH_STENCIL_DESC>());
            }
            let base = GraphicsState::default();
            (blend.AlphaToCoverageEnable, blend.IndependentBlendEnable) =
                (base.blend.AlphaToCoverageEnable, base.blend.IndependentBlendEnable);
            for (t, b) in blend.RenderTarget.iter_mut().zip(&base.blend.RenderTarget) {
                (t.BlendEnable, t.LogicOpEnable, t.SrcBlend, t.DestBlend, t.BlendOp) =
                    (b.BlendEnable, b.LogicOpEnable, b.SrcBlend, b.DestBlend, b.BlendOp);
                (t.SrcBlendAlpha, t.DestBlendAlpha, t.BlendOpAlpha, t.LogicOp) =
                    (b.SrcBlendAlpha, b.DestBlendAlpha, b.BlendOpAlpha, b.LogicOp);
                t.RenderTargetWriteMask = b.RenderTargetWriteMask;
            }
            let d = base.depth;
            (depth.DepthEnable, depth.DepthWriteMask, depth.DepthFunc) =
                (d.DepthEnable, d.DepthWriteMask, d.DepthFunc);
            (depth.StencilEnable, depth.StencilReadMask, depth.StencilWriteMask) =
                (d.StencilEnable, d.StencilReadMask, d.StencilWriteMask);
            (depth.FrontFace, depth.BackFace) = (d.FrontFace, d.BackFace);
            zero_padding(&mut blend, &mut depth);
            (bytes(&blend).to_vec(), bytes(&depth).to_vec())
        };
        assert_eq!(states(0x00), states(0xab));
        assert_eq!(states(0x00), states(0xff));
    }

    #[test]
    fn blending_is_off_for_integer_targets_and_kept_for_the_others() {
        let blending = |g: &GraphicsState| {
            g.blend.RenderTarget.map(|t| t.BlendEnable.as_bool())[..3].to_vec()
        };
        let title = |independent: bool| {
            let mut g = GraphicsState::default();
            g.targets.NumRenderTargets = 3;
            g.targets.RTFormats[..3].copy_from_slice(&[
                DXGI_FORMAT_R32_FLOAT,
                DXGI_FORMAT_B4G4R4A4_UNORM,
                DXGI_FORMAT_R8_UINT,
            ]);
            g.blend.IndependentBlendEnable = independent.into();
            let states = if independent { 3 } else { 1 };
            for t in &mut g.blend.RenderTarget[..states] {
                (t.BlendEnable, t.DestBlend, t.BlendOp) =
                    (true.into(), D3D12_BLEND_ONE, D3D12_BLEND_OP_MAX);
            }
            g
        };
        // A state for each target: the integer one stops blending.
        let mut g = title(true);
        g.settle();
        assert_eq!(blending(&g), [true, true, false]);
        // One state for all of them: each gets its own, the first's.
        let mut g = title(false);
        g.settle();
        assert!(g.blend.IndependentBlendEnable.as_bool());
        assert_eq!(blending(&g), [true, true, false]);
        assert_eq!(g.blend.RenderTarget[1].BlendOp, D3D12_BLEND_OP_MAX);
        // Nothing to do: the state is left as the title gave it.
        let mut g = title(false);
        g.targets.RTFormats[2] = DXGI_FORMAT_R8_UNORM;
        let before = g.describe();
        g.settle();
        assert_eq!(g.describe(), before);
        // A target past the count is not looked at.
        let mut g = title(true);
        g.targets.NumRenderTargets = 2;
        let before = g.describe();
        g.settle();
        assert_eq!(g.describe(), before);
    }

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A pipeline with the depth bounds test is made, and loads again.
    #[test]
    fn a_pipeline_with_depth_bounds_is_made() {
        let Some(host) = host().filter(|_| super::depth_bounds_test()) else {
            eprintln!("no host depth bounds test; skipped");
            return;
        };
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let source = b"float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, id, 0.5, 1); }";
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: the arguments are live and `blob` is the out pointer.
        unsafe {
            D3DCompile(source.as_ptr().cast(), source.len(), PCSTR::null(), None, None, s!("main"), s!("vs_5_1"), 0, 0, &mut blob, None)
        }
        .expect("compiles");
        let blob = blob.expect("code");
        // SAFETY: a live blob.
        let vs = unsafe {
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: blob.GetBufferPointer(),
                BytecodeLength: blob.GetBufferSize(),
            }
        };
        let none = D3D12_SHADER_BYTECODE::default;
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        let plain = g.describe();
        g.bounds = true;
        assert_ne!(plain, g.describe());
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        let name = super::super::library::name("bounds test", &root_blob, &[&code_bytes(&vs)], &g.describe());
        for _ in 0..2 {
            bounds_pipeline(&host.device, &root, [vs, none(), none(), none()], &g, &name).expect("made");
        }
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shader support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        // This pipeline has only a color target, with no depth buffer.
        g.depth.DepthEnable = false.into();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, Some(&name)).expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, Some(&name))
            .expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }

    /// A refused mesh pipeline names the part of its state the driver
    /// refuses: each case breaks one part, which the steps before its own
    /// leave as it is.
    #[test]
    fn a_refused_mesh_pipeline_names_the_part_refused() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shader support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, _) = root_signature(&host.device, &[], &[], false).unwrap();
        let title = || {
            let mut g = GraphicsState::default();
            g.targets.NumRenderTargets = 1;
            g.targets.RTFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            g.depth_format = DXGI_FORMAT_D32_FLOAT;
            g
        };
        type Breaks = fn(&mut GraphicsState);
        let cases: [(&str, Breaks); 4] = [
            ("the blend state or sample mask", |g| {
                // Logic operations are for integer targets only.
                g.blend.RenderTarget[0].LogicOpEnable = true.into();
                g.blend.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_CLEAR;
            }),
            ("the rasterizer state", |g| {
                g.rasterizer.FillMode = D3D12_FILL_MODE(0)
            }),
            ("the depth and stencil state", |g| {
                g.depth.DepthFunc = D3D12_COMPARISON_FUNC(0)
            }),
            ("the target formats, depth format or sample count", |g| {
                g.samples.Count = 0
            }),
        ];
        for (part, breaks) in cases {
            let mut g = title();
            mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, None)
                .expect("the unbroken state is accepted");
            breaks(&mut g);
            assert!(
                mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, None).is_err(),
                "{part}: the broken state was accepted"
            );
            assert_eq!(
                mesh_refusal(&host.device, &root, code(&ms), code(&ps), &g),
                part
            );
        }
        // Accepted only without its pixel shader: one that reads an input
        // the mesh shader does not write.
        let reads = compile(
            &dxc,
            dir.path(),
            "float4 main(float4 c : COLOR) : SV_Target { return c; }",
            "ps_6_5",
        );
        assert_eq!(
            mesh_refusal(&host.device, &root, code(&ms), code(&reads), &title()),
            "the pixel shader or how it links to the mesh shader"
        );
    }
}
