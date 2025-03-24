use crate::nn::model::{Model, ModelConfig};
use burn::nn::loss::Reduction::Mean;
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::{Backend, Config, Int, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

impl<B: AutodiffBackend> Model<B> {
    pub fn train_step(
        self,
        states: Tensor<B, 3, Int>,
        target_actions: Tensor<B, 1, Int>,
        target_values: Tensor<B, 2>,
        optimizer: &mut OptimizerAdaptor<Adam, Model<B>, B>,
        learning_rate: f64,
    ) -> Self {
        // need to seed backend somewhere, probably not here
        // B::seed(config.seed);
        let (predicted_actions, predicted_values) = self.forward(states);
        let loss1 = CrossEntropyLossConfig::new()
            .init(&predicted_actions.device())
            .forward(predicted_actions.clone(), target_actions.clone());
        let mse = MseLoss::new();
        let loss2 = mse.forward(predicted_values, target_values, Mean);
        let loss = loss1 + loss2;

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &self);
        // Update the model using the optimizer.
        optimizer.step(learning_rate, self, grads)
    }
}
