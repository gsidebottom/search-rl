use burn::backend::Wgpu;
use search_rl::nn::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(
        11, 11,
        3,
        121,
    0.5).init::<MyBackend>(&device);

    println!("{}", model);
    
}