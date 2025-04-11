use std::{
    collections::HashMap,
    mem,
    sync::{atomic::AtomicU32, Arc},
};

use image::GenericImageView;
use log::{info, warn};
use wgpu::{util::DeviceExt, RenderPass, SurfaceError};

// Main Thread only
pub struct Display {
    sprite_shader: wgpu::ShaderModule,

    config: wgpu::SurfaceConfiguration,

    pub renderer: Option<Renderer>,

    // This needs to be after renderer, so it is dropped after it (since the Renderer includes a view into the Surface Texture)
    surface: wgpu::Surface<'static>,
}

impl Display {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn from_winit(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();

        assert!(size.width > 0, "Width of window must be greater than 0");
        assert!(size.height > 0, "Height of window must be greater than 0");

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        // TODO: For now we just block on this future, maybe more efficient in the future to have actual async?
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        info!("Using adapter {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None, // Trace path
        ))
        .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),            // 1.
                buffers: &[vertex_desc(), index_desc()], // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None,     // 6.
        });

        surface.configure(&device, &config);

        let output = surface.get_current_texture().unwrap();

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&[
                0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 3, 4, 5]),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            surface,
            config,
            renderer: Some(Renderer {
                queue,
                device,
                encoder,
                output: Some(output),
                pipeline: render_pipeline,
                vertex_buffer,
                index_buffer,
                texture_bind_group_layout,
                textures: HashMap::new(),
            }),

            sprite_shader: shader,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;

            let Renderer {
                queue,
                device,
                output,
                encoder,
                pipeline,
                vertex_buffer,
                index_buffer,
                texture_bind_group_layout,
                textures,
            } = self.renderer.take().unwrap();

            mem::drop(output);

            let output = loop {
                self.surface.configure(&device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(v) => break v,
                    Err(_) => continue,
                }
            };

            self.renderer = Some(Renderer {
                device,
                queue,
                output: Some(output),
                encoder,
                pipeline,
                vertex_buffer,
                index_buffer,
                texture_bind_group_layout,
                textures,
            });
        }
    }

    pub fn get_renderer(&mut self) -> &mut Renderer {
        let Some(renderer) = self.renderer.as_mut() else {
            unreachable!()
        };

        renderer
    }

    pub fn finish_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(renderer) = self.renderer.take() else {
            unreachable!();
        };

        let Renderer {
            queue,
            device,
            output,
            encoder,
            pipeline,
            vertex_buffer,
            index_buffer,
            texture_bind_group_layout,
            textures,
        } = renderer;

        if let Some(output) = output {
            // submit will accept anything that implements IntoIter
            queue.submit(std::iter::once(encoder.finish()));
            output.present();
        } else {
            warn!("Could not finish frame, due to missing output!")
        }

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let output = self.surface.get_current_texture().ok();

        self.renderer = Some(Renderer {
            device,
            queue,
            output,
            encoder,
            pipeline,
            vertex_buffer,
            index_buffer,
            texture_bind_group_layout,
            textures,
        });

        Ok(())
    }

    // fn create_texture() -> Texture {}
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    output: Option<wgpu::SurfaceTexture>,
    encoder: wgpu::CommandEncoder,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    texture_bind_group_layout: wgpu::BindGroupLayout,

    textures: HashMap<u32, wgpu::BindGroup>,
}

impl RendererTrait for Renderer {
    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, layer: &Layer) {
        let diffuse_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let Some(output) = &self.output else {
            return;
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut render_pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("layer renderer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        // For each texture:
        for (texture_id, (sprite, instances)) in &layer.commands.commands {
            let bind_group = if let Some(texture) = self.textures.get(texture_id) {
                texture
            } else {
                info!("Uploading texture {} to gpu", texture_id);
                // Upload that texture:
                let texture_size = wgpu::Extent3d {
                    width: sprite.texture.dims.0,
                    height: sprite.texture.dims.1,
                    // All textures are stored as 3D, we represent our 2D texture
                    // by setting depth to 1.
                    depth_or_array_layers: 1,
                };

                let diffuse_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    size: texture_size,
                    mip_level_count: 1, // We'll talk about this a little later
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // Most images are stored using sRGB, so we need to reflect that here.
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                    // COPY_DST means that we want to copy data to this texture
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    label: Some("diffuse_texture"),
                    // This is the same as with the SurfaceConfig. It
                    // specifies what texture formats can be used to
                    // create TextureViews for this texture. The base
                    // texture format (Rgba8UnormSrgb in this case) is
                    // always supported. Note that using a different
                    // texture format is not supported on the WebGL2
                    // backend.
                    view_formats: &[],
                });

                self.queue.write_texture(
                    // Tells wgpu where to copy the pixel data
                    wgpu::ImageCopyTexture {
                        texture: &diffuse_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    // The actual pixel data
                    &sprite.texture.data,
                    // The layout of the texture
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * sprite.texture.dims.0),
                        rows_per_image: Some(sprite.texture.dims.1),
                    },
                    texture_size,
                );

                // We don't need to configure the texture view much, so let's
                // let wgpu define it.
                let diffuse_texture_view =
                    diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let diffuse_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &self.texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                            },
                        ],
                        label: Some("diffuse_bind_group"),
                    });

                self.textures.insert(*texture_id, diffuse_bind_group);
                let Some(ret) = self.textures.get(texture_id) else {
                    unreachable!();
                };

                ret
            };

            render_pass.set_pipeline(&self.pipeline);

            let instance_data = &instances;
            let instance_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer for layer"),
                        contents: bytemuck::cast_slice(instance_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.set_bind_group(0, bind_group, &[]);

            render_pass.draw_indexed(0..6, 0, 0..instances.len().try_into().unwrap());
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawRenderer {
    queue: wgpu::Queue,
    device: wgpu::Device,

    render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl RawRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),            // 1.
                buffers: &[vertex_desc(), index_desc()], // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None,     // 6.
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&[
                0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 3, 4, 5]),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            queue: queue.clone(),
            device: device.clone(),

            texture_bind_group_layout,

            render_pipeline,

            vertex_buffer,
            index_buffer,
        }
    }

    pub fn start_draw<'a, 'b, 'c>(
        &'a self,
        render_pass: &'b mut RenderPass<'c>,
    ) -> InprogressRawRenderer<'a, 'b, 'c> {
        InprogressRawRenderer {
            raw_renderer: self,
            render_pass,
        }
    }
}

pub struct InprogressRawRenderer<'a, 'b, 'c> {
    raw_renderer: &'a RawRenderer,
    render_pass: &'b mut RenderPass<'c>,
}

impl<'a, 'b, 'c> RendererTrait for InprogressRawRenderer<'a, 'b, 'c> {
    fn draw(&mut self, layer: &Layer) {
        let diffuse_sampler = self
            .raw_renderer
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        self.render_pass
            .set_vertex_buffer(0, self.raw_renderer.vertex_buffer.slice(..));

        // For each texture:
        for (texture_id, (sprite, instances)) in &layer.commands.commands {
            let bind_group = {
                info!("Uploading texture {} to gpu", texture_id);
                // Upload that texture:
                let texture_size = wgpu::Extent3d {
                    width: sprite.texture.dims.0,
                    height: sprite.texture.dims.1,
                    // All textures are stored as 3D, we represent our 2D texture
                    // by setting depth to 1.
                    depth_or_array_layers: 1,
                };

                let diffuse_texture =
                    self.raw_renderer
                        .device
                        .create_texture(&wgpu::TextureDescriptor {
                            size: texture_size,
                            mip_level_count: 1, // We'll talk about this a little later
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            // Most images are stored using sRGB, so we need to reflect that here.
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                            // COPY_DST means that we want to copy data to this texture
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            label: Some("diffuse_texture"),
                            // This is the same as with the SurfaceConfig. It
                            // specifies what texture formats can be used to
                            // create TextureViews for this texture. The base
                            // texture format (Rgba8UnormSrgb in this case) is
                            // always supported. Note that using a different
                            // texture format is not supported on the WebGL2
                            // backend.
                            view_formats: &[],
                        });

                self.raw_renderer.queue.write_texture(
                    // Tells wgpu where to copy the pixel data
                    wgpu::ImageCopyTexture {
                        texture: &diffuse_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    // The actual pixel data
                    &sprite.texture.data,
                    // The layout of the texture
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * sprite.texture.dims.0),
                        rows_per_image: Some(sprite.texture.dims.1),
                    },
                    texture_size,
                );

                // We don't need to configure the texture view much, so let's
                // let wgpu define it.
                let diffuse_texture_view =
                    diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let diffuse_bind_group =
                    self.raw_renderer
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.raw_renderer.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &diffuse_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                                },
                            ],
                            label: Some("diffuse_bind_group"),
                        });

                diffuse_bind_group
            };

            self.render_pass
                .set_pipeline(&self.raw_renderer.render_pipeline);

            let instance_data = &instances;
            let instance_buffer =
                self.raw_renderer
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer for layer"),
                        contents: bytemuck::cast_slice(instance_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            self.render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            self.render_pass.set_index_buffer(
                self.raw_renderer.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            self.render_pass.set_bind_group(0, &bind_group, &[]);

            self.render_pass
                .draw_indexed(0..6, 0, 0..instances.len().try_into().unwrap());
        }
    }
}

pub trait RendererTrait {
    fn draw(&mut self, layer: &Layer);
}

#[derive(Debug)]
pub struct Layer {
    x_mult: f32,
    y_mult: f32,
    commands: DrawSpriteCommands,
}

enum LayerType {
    Transparent,
    Oquaque(wgpu::Color),
}

#[derive(Debug, Default)]
struct DrawSpriteCommands {
    commands: HashMap<u32, (Sprite, Vec<Instance>)>,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Instance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub width_mul: f32,
    pub width_offs: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DrawInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub animation_frame: u32,
}

impl Layer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            x_mult: 1.0,
            y_mult: 1.0,
            commands: DrawSpriteCommands::default(),
        }
    }

    #[must_use]
    pub fn square_tile_grid(tile_size: f32) -> Self {
        // TODO: The tiles are not square if the draw surface is not square
        Self {
            x_mult: tile_size,
            y_mult: tile_size,
            commands: DrawSpriteCommands::default(),
        }
    }

    /// # Panics
    /// If the instances animation frame is not in range for the sprites' texture
    pub fn draw_sprite(&mut self, sprite: &Sprite, instance: DrawInstance) {
        assert!(instance.animation_frame < sprite.texture.number_anim_frames);

        if let Some((a, b)) = self.commands.commands.get_mut(&sprite.texture.id) {
            b.push(Instance {
                position: [
                    instance.position[0] * self.x_mult,
                    instance.position[1] * self.y_mult,
                ],
                size: [
                    instance.size[0] * self.x_mult,
                    instance.size[1] * self.y_mult,
                ],
                #[allow(clippy::cast_precision_loss)]
                width_mul: 1.0 / sprite.texture.number_anim_frames as f32,
                #[allow(clippy::cast_precision_loss)]
                width_offs: instance.animation_frame as f32
                    / sprite.texture.number_anim_frames as f32,
            });
        } else {
            self.commands.commands.insert(
                sprite.texture.id,
                (
                    sprite.clone(),
                    vec![Instance {
                        position: [
                            instance.position[0] * self.x_mult,
                            instance.position[1] * self.y_mult,
                        ],
                        size: [
                            instance.size[0] * self.x_mult,
                            instance.size[1] * self.y_mult,
                        ],
                        #[allow(clippy::cast_precision_loss)]
                        width_mul: 1.0 / sprite.texture.number_anim_frames as f32,
                        #[allow(clippy::cast_precision_loss)]
                        width_offs: instance.animation_frame as f32
                            / sprite.texture.number_anim_frames as f32,
                    }],
                ),
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sprite {
    texture: Texture,
    custom_shader: Option<Arc<Shader>>,
}

#[derive(Debug)]
pub struct Shader {}

struct TextureAtlas {}

impl Sprite {
    #[must_use]
    pub fn new(texture: Texture) -> Self {
        Self {
            texture,
            custom_shader: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Texture {
    id: u32,
    number_anim_frames: u32,
    data: Arc<[u8]>,
    dims: (u32, u32),
}

impl Default for Texture {
    fn default() -> Self {
        let diffuse_bytes = include_bytes!("image.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8().into_vec();

        let dimensions = diffuse_image.dimensions();

        Self::new(1, diffuse_rgba, dimensions)
    }
}

impl Texture {
    pub fn new(number_frames: u32, data: impl Into<Arc<[u8]>>, dims: (u32, u32)) -> Self {
        static TEXTURE_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
        Self {
            id: TEXTURE_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            number_anim_frames: number_frames,
            data: data.into(),
            dims,
        }
    }
}

fn vertex_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: (2 * std::mem::size_of::<f32>()).try_into().unwrap(), // 1.
        step_mode: wgpu::VertexStepMode::Vertex,                            // 2.
        attributes: &[
            // 3.
            wgpu::VertexAttribute {
                offset: 0,                             // 4.
                shader_location: 0,                    // 5.
                format: wgpu::VertexFormat::Float32x2, // 6.
            },
        ],
    }
}

fn index_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Instance>().try_into().unwrap(), // 1.
        step_mode: wgpu::VertexStepMode::Instance,                         // 2.
        attributes: &[
            // 3.
            wgpu::VertexAttribute {
                offset: 0,                             // 4.
                shader_location: 1,                    // 5.
                format: wgpu::VertexFormat::Float32x2, // 6.
            },
            wgpu::VertexAttribute {
                offset: 2u64 * std::mem::size_of::<f32>() as u64, // 4.
                shader_location: 2,                               // 5.
                format: wgpu::VertexFormat::Float32x2,            // 6.
            },
            wgpu::VertexAttribute {
                offset: 4 * std::mem::size_of::<f32>() as u64,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 5 * std::mem::size_of::<f32>() as u64,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
        ],
    }
}
