use wgpu::util::DeviceExt;
use bytemuck:: {Pod, Zeroable, cast_slice};
use cgmath::{ Matrix, SquareMatrix };
use rand::Rng;
use wgpu_simplified as ws;
use super::vertex_data as vd;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub fn cube_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let(pos, _, norm, _, ind, _) = vd::create_cube_data(2.0);
    let mut data:Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex{position: pos[i], normal: norm[i]});
    }
    (data.to_vec(), ind)
}

pub fn sphere_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let (pos, norm, _, ind, _) = vd::create_sphere_data(2.2, 20, 30);
    let mut data:Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex{position: pos[i], normal: norm[i]});
    }
    (data.to_vec(), ind)
}

pub fn torus_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let (pos, norm, ind, _) = vd::create_torus_data(1.8, 0.4, 60, 20);
    let mut data:Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex{position: pos[i], normal: norm[i]});
    }
    (data.to_vec(), ind)
}

pub fn create_object_buffers(device: &wgpu::Device) -> (Vec<wgpu::Buffer>, Vec<wgpu::Buffer>, Vec<u32>) {
    let (cube_vertex_data, cube_index_data) = cube_vertices();
    let (sphere_vertex_data, sphere_index_data) = sphere_vertices();
    let (torus_vertex_data, torus_index_data) = torus_vertices();

    let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Vertex Buffer"),
        contents: cast_slice(&cube_vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("Cube Index Buffer"),
        contents: cast_slice(&cube_index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere Vertex Buffer"),
        contents: cast_slice(&sphere_vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("Sphere Index Buffer"),
        contents: cast_slice(&sphere_index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let torus_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Torus Vertex Buffer"),
        contents: cast_slice(&torus_vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let torus_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("Torus Index Buffer"),
        contents: cast_slice(&torus_index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    (
        vec![cube_vertex_buffer, sphere_vertex_buffer, torus_vertex_buffer],
        vec![cube_index_buffer, sphere_index_buffer, torus_index_buffer],
        vec![cube_index_data.len() as u32, sphere_index_data.len() as u32, torus_index_data.len() as u32]
    )
}

#[allow(dead_code)]
pub fn create_transform_mat_color(objects_count:u32, translate_default:bool) -> (Vec<[f32; 16]>, Vec<[f32; 16]>, Vec<[f32; 4]>){
    let mut model_mat:Vec<[f32; 16]> = vec![];
    let mut normal_mat:Vec<[f32; 16]> = vec![];
    let mut color_vec:Vec<[f32; 4]> = vec![];

    for _i in 0..objects_count {
        let mut rng = rand::thread_rng();
        let mut translation = [rng.gen::<f32>() * 60.0 - 53.0, rng.gen::<f32>() * 50.0 - 45.0, -15.0 - rng.gen::<f32>() * 50.0];
        if !translate_default {
            translation = [rng.gen::<f32>() * 50.0 - 25.0, rng.gen::<f32>() * 40.0 - 18.0, -30.0 - rng.gen::<f32>() * 50.0];
        }
        let rotation = [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()];
        let scale = [1.0, 1.0, 1.0];
        let m = ws::create_model_mat(translation, rotation, scale);
        let n = (m.invert().unwrap()).transpose();
        let color = [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>(), 1.0];
        model_mat.push(*(m.as_ref()));
        normal_mat.push(*(n.as_ref()));
        color_vec.push(color);
    }

    (model_mat, normal_mat, color_vec)
}