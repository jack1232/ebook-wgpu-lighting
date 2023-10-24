use wgpu_lighting:: vertex_data as vd;
mod common_blinn_phong;
use common_blinn_phong::Vertex;

fn create_vertices() -> (Vec<Vertex>, Vec<u16>, Vec<u16>) {
    let (pos, norm, _, ind, ind2) = vd::create_sphere_data(2.2, 20, 30);
    let mut data:Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex{position: pos[i], normal: norm[i]});
    }
    (data.to_vec(), ind, ind2)
}

fn main(){
    let mut sample_count = 1 as u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }

    let (vertex_data, index_data, index_data2) = create_vertices();
    common_blinn_phong::run(&vertex_data, &index_data, &index_data2, sample_count, "blinn_phong_sphere");
}