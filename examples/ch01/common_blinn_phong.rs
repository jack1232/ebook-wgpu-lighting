use bytemuck::{cast_slice, Pod, Zeroable};
use cgmath::{Matrix, Matrix4, SquareMatrix};
use rand;
use std::{iter, mem};
use wgpu::{util::DeviceExt, VertexBufferLayout};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct IMaterial {
    ambient_intensity: f32,
    diffuse_intensity: f32,
    specular_intensity: f32,
    specular_shininess: f32,
}

impl Default for IMaterial {
    fn default() -> IMaterial {
        IMaterial {
            ambient_intensity: 0.2,
            diffuse_intensity: 0.8,
            specular_intensity: 0.4,
            specular_shininess: 30.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

struct State {
    init: ws::IWgpuInit,
    pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffer: wgpu::Buffer,
    index_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    indices_lens: Vec<u32>,
    plot_type: u32,
    rotation_speed: f32,

    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
}

impl State {
    async fn new(
        window: &Window,
        vertex_data: &Vec<Vertex>,
        index_data: &Vec<u16>,
        index_data2: &Vec<u16>,
        sample_count: u32,
    ) -> Self {
        let init = ws::IWgpuInit::new(&window, sample_count, None).await;

        let vs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shader_vert.wgsl"));
        let fs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("blinn_phong_frag.wgsl"));

        // uniform data
        let camera_position = (3.0, 1.5, 3.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();

        let (view_mat, project_mat, _) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        // create vertex uniform buffers

        // model_mat and vp_mat will be stored in vertex_uniform_buffer inside the update function
        let vert_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Uniform Buffer"),
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // create light uniform buffer. here we set eye_position = camera_position and light_position = eye_position
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_uniform_buffer2 = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer 2"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_position: &[f32; 3] = camera_position.as_ref();
        let eye_position: &[f32; 3] = camera_position.as_ref();
        init.queue
            .write_buffer(&light_uniform_buffer, 0, cast_slice(light_position));
        init.queue
            .write_buffer(&light_uniform_buffer, 16, cast_slice(eye_position));
        init.queue
            .write_buffer(&light_uniform_buffer2, 0, cast_slice(light_position));
        init.queue
            .write_buffer(&light_uniform_buffer2, 16, cast_slice(eye_position));

        // set specular light color to white
        let specular_color: [f32; 3] = [1.0, 1.0, 1.0];
        init.queue.write_buffer(
            &light_uniform_buffer,
            48,
            cast_slice(specular_color.as_ref()),
        );
        init.queue.write_buffer(
            &light_uniform_buffer2,
            48,
            cast_slice(specular_color.as_ref()),
        );

        // set default object color to red:
        let object_color: [f32; 3] = [1.0, 0.0, 0.0];
        init.queue
            .write_buffer(&light_uniform_buffer, 32, cast_slice(object_color.as_ref()));

        // set default wireframe color to yellow:
        let wireframe_color: [f32; 3] = [1.0, 1.0, 0.0];
        init.queue.write_buffer(
            &light_uniform_buffer2,
            32,
            cast_slice(wireframe_color.as_ref()),
        );

        // material uniform buffer
        let material_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // set default material parameters
        let material = [0.2 as f32, 0.8, 0.4, 30.0];
        init.queue
            .write_buffer(&material_uniform_buffer, 0, cast_slice(material.as_ref()));

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );
        let (vert_bind_group_layout2, vert_bind_group2) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );

        // uniform bind group for fragment shader
        let (frag_bind_group_layout, frag_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT],
            &[
                light_uniform_buffer.as_entire_binding(),
                material_uniform_buffer.as_entire_binding(),
            ],
        );
        let (frag_bind_group_layout2, frag_bind_group2) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT],
            &[
                light_uniform_buffer2.as_entire_binding(),
                material_uniform_buffer.as_entire_binding(),
            ],
        );

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&vert_bind_group_layout, &frag_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mut ppl = ws::IRenderPipeline {
            vs_shader: Some(&vs_shader),
            fs_shader: Some(&fs_shader),
            pipeline_layout: Some(&pipeline_layout),
            vertex_buffer_layout: &[vertex_buffer_layout],
            ..Default::default()
        };
        let pipeline = ppl.new(&init);

        let vertex_buffer_layout2 = VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let pipeline_layout2 =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout 2"),
                    bind_group_layouts: &[&vert_bind_group_layout2, &frag_bind_group_layout2],
                    push_constant_ranges: &[],
                });

        let mut ppl2 = ws::IRenderPipeline {
            topology: wgpu::PrimitiveTopology::LineList,
            vs_shader: Some(&vs_shader),
            fs_shader: Some(&fs_shader),
            pipeline_layout: Some(&pipeline_layout2),
            vertex_buffer_layout: &[vertex_buffer_layout2],
            ..Default::default()
        };
        let pipeline2 = ppl2.new(&init);

        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        let vertex_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: cast_slice(vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

        let index_buffer2 = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer 2"),
                contents: bytemuck::cast_slice(index_data2),
                usage: wgpu::BufferUsages::INDEX,
            });

        Self {
            init,
            pipelines: vec![pipeline, pipeline2],
            vertex_buffer,
            index_buffers: vec![index_buffer, index_buffer2],
            uniform_bind_groups: vec![
                vert_bind_group,
                frag_bind_group,
                vert_bind_group2,
                frag_bind_group2,
            ],
            uniform_buffers: vec![
                vert_uniform_buffer,
                light_uniform_buffer,
                material_uniform_buffer,
                light_uniform_buffer2,
            ],
            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,
            indices_lens: vec![index_data.len() as u32, index_data2.len() as u32],
            plot_type: 0,
            rotation_speed: 1.0,

            ambient: material[0],
            diffuse: material[1],
            specular: material[2],
            shininess: material[3],
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat =
                ws::create_projection_mat(new_size.width as f32 / new_size.height as f32, true);
            self.depth_texture_view = ws::create_depth_view(&self.init);
            if self.init.sample_count > 1 {
                self.msaa_texture_view = ws::create_msaa_texture_view(&self.init);
            }
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                VirtualKeyCode::LControl => {
                    let scolor: [f32; 3] = [rand::random(), rand::random(), rand::random()];
                    self.init.queue.write_buffer(
                        &self.uniform_buffers[1],
                        32,
                        cast_slice(scolor.as_ref()),
                    );
                    true
                }
                VirtualKeyCode::LAlt => {
                    let wcolor: [f32; 3] = [rand::random(), rand::random(), rand::random()];
                    self.init.queue.write_buffer(
                        &self.uniform_buffers[3],
                        32,
                        cast_slice(wcolor.as_ref()),
                    );
                    true
                }
                VirtualKeyCode::Space => {
                    self.plot_type = (self.plot_type + 1) % 3;
                    true
                }
                VirtualKeyCode::Q => {
                    self.ambient += 0.01;
                    println!("ambient intensity = {}", self.ambient);
                    true
                }
                VirtualKeyCode::A => {
                    self.ambient -= 0.05;
                    if self.ambient < 0.0 {
                        self.ambient = 0.0;
                    }
                    println!("ambient intensity = {}", self.ambient);
                    true
                }
                VirtualKeyCode::W => {
                    self.diffuse += 0.05;
                    println!("diffuse intensity = {}", self.diffuse);
                    true
                }
                VirtualKeyCode::S => {
                    self.diffuse -= 0.05;
                    if self.diffuse < 0.0 {
                        self.diffuse = 0.0;
                    }
                    println!("diffuse intensity = {}", self.diffuse);
                    true
                }
                VirtualKeyCode::E => {
                    self.specular += 0.05;
                    println!("specular intensity = {}", self.specular);
                    true
                }
                VirtualKeyCode::D => {
                    self.specular -= 0.05;
                    if self.specular < 0.0 {
                        self.specular = 0.0;
                    }
                    println!("specular intensity = {}", self.specular);
                    true
                }
                VirtualKeyCode::R => {
                    self.shininess += 5.0;
                    println!("specular shininess = {}", self.shininess);
                    true
                }
                VirtualKeyCode::F => {
                    self.shininess -= 5.0;
                    if self.shininess < 0.0 {
                        self.shininess = 0.0;
                    }
                    println!("specular shininess = {}", self.shininess);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // update uniform buffer
        let dt = self.rotation_speed * dt.as_secs_f32();
        let model_mat =
            ws::create_model_mat([0.0, 0.0, 0.0], [dt.sin(), dt.cos(), 0.0], [1.0, 1.0, 1.0]);
        let view_project_mat = self.project_mat * self.view_mat;

        let normal_mat = (model_mat.invert().unwrap()).transpose();

        let model_ref: &[f32; 16] = model_mat.as_ref();
        let view_projection_ref: &[f32; 16] = view_project_mat.as_ref();
        let normal_ref: &[f32; 16] = normal_mat.as_ref();

        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 0, cast_slice(view_projection_ref));
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 64, cast_slice(model_ref));
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 128, cast_slice(normal_ref));

        // update material
        let material = [self.ambient, self.diffuse, self.specular, self.shininess];
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[2], 0, cast_slice(material.as_ref()));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        //let output = self.init.surface.get_current_frame()?.output;
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let color_attach = ws::create_color_attachment(&view);
            let msaa_attach = ws::create_msaa_color_attachment(&view, &self.msaa_texture_view);
            let color_attachment = if self.init.sample_count == 1 {
                color_attach
            } else {
                msaa_attach
            };
            let depth_attachment = ws::create_depth_stencil_attachment(&self.depth_texture_view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
            });

            let plot_type = if self.plot_type == 1 {
                "shape_only"
            } else if self.plot_type == 2 {
                "wireframe_only"
            } else {
                "both"
            };

            if plot_type == "shape_only" || plot_type == "both" {
                render_pass.set_pipeline(&self.pipelines[0]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint16);
                render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
                render_pass.set_bind_group(1, &self.uniform_bind_groups[1], &[]);
                render_pass.draw_indexed(0..self.indices_lens[0], 0, 0..1);
            }

            if plot_type == "wireframe_only" || plot_type == "both" {
                render_pass.set_pipeline(&self.pipelines[1]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffers[1].slice(..), wgpu::IndexFormat::Uint16);
                render_pass.set_bind_group(0, &self.uniform_bind_groups[2], &[]);
                render_pass.set_bind_group(1, &self.uniform_bind_groups[3], &[]);
                render_pass.draw_indexed(0..self.indices_lens[1], 0, 0..1);
            }
        }

        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn run(
    vertex_data: &Vec<Vertex>,
    index_data: &Vec<u16>,
    index_data2: &Vec<u16>,
    sample_count: u32,
    title: &str,
) {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(title);

    let mut state = pollster::block_on(State::new(
        &window,
        &vertex_data,
        index_data,
        index_data2,
        sample_count,
    ));
    let render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(_) => {
            let now = std::time::Instant::now();
            let dt = now - render_start_time;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}