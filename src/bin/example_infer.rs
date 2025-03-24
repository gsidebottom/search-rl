use std::path::Path;
use search_rl::nn_example::{model::ModelConfig, training::{TrainingConfig, train}, inference::infer};
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    // can load already trained and saved model...
    if !Path::new(&format!("{artifact_dir}/model.mpk")).exists() {
        train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
            device.clone(),
        );
    }
    infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}