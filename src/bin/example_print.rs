use burn::backend::Wgpu;
use burn::tensor::{check_closeness, Tensor};
use search_rl::nn_example::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{}", model);

    let tensor = Tensor::<burn::backend::Autodiff<MyBackend>, 2>::full([2, 3], 0.123456789, &Default::default());
    println!("{}", tensor);
    
    let device = Default::default();
    let tensor1 = Tensor::<MyBackend, 1>::from_floats(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.001, 7.002, 8.003, 9.004, 10.1],
        &device,
    );
    let tensor2 = Tensor::<MyBackend, 1>::from_floats(
        [1.0, 2.0, 3.0, 4.000, 5.0, 6.0, 7.001, 8.002, 9.003, 10.004],
        &device,
    );

    check_closeness(&tensor1, &tensor2);
}