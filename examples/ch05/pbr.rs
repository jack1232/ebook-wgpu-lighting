use bytemuck::cast_slice;
use cgmath::Matrix4;
use std::{iter, mem};
use wgpu::VertexBufferLayout;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;
use wgpu_lighting::common_instance;

const ANIMATION_SPEED: f32 = 1.0;
const NUM_CUBES: u32 = 50;
const NUM_SPHERES: u32 = 50;
const NUM_TORI: u32 = 50;

struct State {
    init: ws::IWgpuInit,
    pipeline: wgpu::RenderPipeline,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    indices_lens: Vec<u32>,

    intensity: f32,
    metallic: f32,
    roughness: f32,
}

impl State {
    async fn new(window: &Window, sample_count: u32) -> Self {
        let init = ws::IWgpuInit::new(&window, sample_count, None).await;

        let vs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("../ch02/shader_instance_vert.wgsl"));
        let fs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("pbr_frag.wgsl"));

        let objects_count = NUM_CUBES + NUM_SPHERES + NUM_TORI;

        // uniform data
        let camera_position = (0.0, 0.0, 1.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();
        let eye_position = camera_position;
        let light_position = camera_position;

        let (model_mat, normal_mat, color_vec) =
            common_instance::create_transform_mat_color(objects_count, false);

        let (view_mat, project_mat, vp_mat) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        // create uniform view-projection buffers
        let vp_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View-Projection Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &vp_uniform_buffer,
            0,
            cast_slice(vp_mat.as_ref() as &[f32; 16]),
        );

        // model uniform buffer
        let model_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Model Uniform Buffer"),
            size: 64 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue
            .write_buffer(&model_uniform_buffer, 0, cast_slice(&model_mat));

        // normal storage buffer
        let normal_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Normal Uniform Buffer"),
            size: 64 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue
            .write_buffer(&normal_uniform_buffer, 0, cast_slice(&normal_mat));

        // color storage buffer
        let color_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("color Uniform Buffer"),
            size: 16 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue
            .write_buffer(&color_uniform_buffer, 0, cast_slice(&color_vec));

        // light uniform buffer.
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &light_uniform_buffer,
            0,
            cast_slice(light_position.as_ref() as &[f32; 3]),
        );

        // material uniform buffer
        let material_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Uniform Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // set default material parameters
        init.queue.write_buffer(
            &material_uniform_buffer,
            0,
            cast_slice(eye_position.as_ref() as &[f32; 3]),
        );
        let material = [0.2 as f32, 0.8];
        init.queue
            .write_buffer(&material_uniform_buffer, 16, cast_slice(material.as_ref()));

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
            ],
            vec![
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
            ],
            &[
                vp_uniform_buffer.as_entire_binding(),
                model_uniform_buffer.as_entire_binding(),
                normal_uniform_buffer.as_entire_binding(),
                color_uniform_buffer.as_entire_binding(),
            ],
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

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<common_instance::Vertex>() as wgpu::BufferAddress,
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

        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        // vertex and index buffers for objects:
        let (vertex_buffers, index_buffers, index_lens) =
            common_instance::create_object_buffers(&init.device);

        Self {
            init,
            pipeline,
            vertex_buffers,
            index_buffers,
            uniform_bind_groups: vec![vert_bind_group, frag_bind_group],
            uniform_buffers: vec![
                vp_uniform_buffer,
                model_uniform_buffer,
                normal_uniform_buffer,
                color_uniform_buffer,
                light_uniform_buffer,
                material_uniform_buffer,
            ],
            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,
            indices_lens: index_lens,

            intensity: 300.0,
            roughness: material[0],
            metallic: material[1],
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
                VirtualKeyCode::Q => {
                    self.intensity += 10.0;
                    println!("intensity = {}", self.intensity);
                    true
                }
                VirtualKeyCode::A => {
                    self.intensity -= 10.0;
                    if self.intensity < 10.0 {
                        self.intensity = 10.0;
                    }
                    println!("intensity = {}", self.intensity);
                    true
                }
                VirtualKeyCode::W => {
                    self.roughness += 0.01;
                    println!("roughness = {}", self.roughness);
                    true
                }
                VirtualKeyCode::S => {
                    self.roughness -= 0.01;
                    if self.roughness < 0.05 {
                        self.roughness = 0.05;
                    }
                    println!("roughness = {}", self.roughness);
                    true
                }
                VirtualKeyCode::E => {
                    self.metallic += 0.01;
                    println!("metallic = {}", self.metallic);
                    true
                }
                VirtualKeyCode::D => {
                    self.metallic -= 0.01;
                    if self.metallic < 0.001 {
                        self.metallic = 0.001;
                    }
                    println!("metallic = {}", self.metallic);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // update uniform buffer
        let dt = ANIMATION_SPEED * dt.as_secs_f32();
        let sn = (2.0 + dt.sin()) / 3.0;
        let cn = (2.0 + dt.cos()) / 3.0;
        let dz = 5.0;
        let factor = 15.0;
        let light_pos0 = [-factor * sn, factor * cn, factor * cn - dz, 1.0];
        let light_pos1 = [factor * sn, factor * cn, factor * cn - dz, 1.0];
        let light_pos2 = [-factor * sn, -factor * cn, factor * cn - dz, 1.0];
        let light_pos3 = [factor * sn, -factor * cn, factor * cn - dz, 1.0];
        let intensity = [self.intensity, self.intensity, self.intensity, 1.0];

        let data = [
            light_pos0, intensity, light_pos1, intensity, light_pos2, intensity, light_pos3,
            intensity,
        ]
        .concat();

        self.init
            .queue
            .write_buffer(&self.uniform_buffers[4], 0, cast_slice(data.as_ref()));

        let view_project_mat = self.project_mat * self.view_mat;
        let view_projection_ref: &[f32; 16] = view_project_mat.as_ref();
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 0, cast_slice(view_projection_ref));

        // update material
        let material = [self.roughness, self.metallic];
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[5], 16, cast_slice(material.as_ref()));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_groups[1], &[]);

            // draw cubes
            render_pass.set_vertex_buffer(0, self.vertex_buffers[0].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.indices_lens[0], 0, 0..NUM_CUBES);

            // draw spheres
            render_pass.set_vertex_buffer(0, self.vertex_buffers[1].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[1].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.indices_lens[1],
                0,
                NUM_CUBES..NUM_CUBES + NUM_SPHERES,
            );

            // draw tori
            render_pass.set_vertex_buffer(0, self.vertex_buffers[2].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[2].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.indices_lens[2],
                0,
                NUM_CUBES + NUM_SPHERES..NUM_CUBES + NUM_SPHERES + NUM_TORI,
            );
        }
        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    let mut sample_count = 1 as u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(&*format!("{}", "pbr_light"));

    let mut state = pollster::block_on(State::new(&window, sample_count));
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