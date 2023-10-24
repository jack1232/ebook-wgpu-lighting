use common_blinn_phong::Vertex;
use wgpu_lighting:: vertex_data as vd;
mod common_blinn_phong;

fn create_vertices() -> (Vec<common_blinn_phong::Vertex>, Vec<u16>, Vec<u16>) {
    let(pos, _, normal, _, ind, ind2) = vd::create_cube_data(2.0);
    let mut data:Vec<common_blinn_phong::Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex{position:pos[i], normal: normal[i]});
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
    common_blinn_phong::run(&vertex_data, &index_data, &index_data2, sample_count, "blinn_phong_cube");
}