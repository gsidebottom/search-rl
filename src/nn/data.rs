use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Tensor};
use burn::tensor::TensorData;
use itertools::{repeat_n, Itertools};
use crate::mcts::Example;

#[derive(Clone)]
pub struct RLSearchBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> RLSearchBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct RLSearchBatch<B: Backend> {
    pub states: Tensor<B, 3>,   // batch_size x height x width
    pub pis: Tensor<B, 1>, // batch_size x action_count
    pub values: Tensor<B, 1>,   // batch_size x 1
}


impl<B: Backend, const N: usize, const D: usize> Batcher<Example<N, D>, RLSearchBatch<B>> for RLSearchBatcher<B> {
    fn batch(&self, examples: Vec<Example<N, D>>) -> RLSearchBatch<B> {
       let states = examples.iter().map(|example| TensorData::from(example.state))
           .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
           .map(|tensor| tensor.reshape([1, D, D]))
           .collect_vec();
       let states = Tensor::cat(states, 0).to_device(&self.device);

        let pis = examples.iter().map(|example| {
            let pi_count = example.pi.len();
            let pis = repeat_n(0.0, pi_count-N.max(pi_count))
                .chain(example.pi.iter()
                    .map(|f| f.0))
                .collect_array::<N>()
                .unwrap();
            TensorData::from(pis)
        })
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .collect_vec();
        let pis = Tensor::cat(pis, 0).to_device(&self.device);

        let values = examples.iter().map(|example| TensorData::from([example.value]))
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .collect_vec();
        let values = Tensor::cat(values, 0).to_device(&self.device);
        RLSearchBatch {
            states,
            pis,
            values,
        }
    }
}