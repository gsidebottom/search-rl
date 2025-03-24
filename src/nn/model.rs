use burn::tensor::activation::{softmax, tanh};
use burn::{
    nn::{
        conv::{Conv1d, Conv1dConfig}
        ,
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear,
        LinearConfig, PaddingConfig1d, Relu
    },
    prelude::*,
};

/// Keras model
/// def __init__(self, game, args):
///     self.board_x, self.board_y = game.getBoardSize()
///     self.action_size = game.getActionSize()
///     self.args = args
///
///     self.input_layer = Input(shape=(self.board_x, self.board_y))
///
///     h_conv1 = Activation('relu')(BatchNormalization(axis=2)(Conv1D(args.num_channels, 3, padding='same', use_bias=False)(self.input_layer)))
///     h_conv1_flat = Flatten()(h_conv1)
///
///     s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(16, use_bias=False)(h_conv1_flat))))
///     s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(16, use_bias=False)(s_fc1))))
///
///     self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
///     self.v = Dens(1, activation='tanh', name='v')(s_fc2)
///
///     self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
/// ChatGPT conversion to burn <https://chatgpt.com/share/67d616f0-98c0-800b-b0b3-b014caa513ba>

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // tangled
    conv1: Conv1d<B>,
    batch_norm1: BatchNorm<B, 2>,
    fc1: Linear<B>,
    batch_norm2: BatchNorm<B, 1>,
    fc2: Linear<B>,
    batch_norm3: BatchNorm<B, 1>,
    dropout: Dropout,
    pi: Linear<B>,
    v: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    board_x: usize,
    board_y: usize,
    num_channels: usize,
    action_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
    lr: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // Conv1D(args.num_channels, 3, padding=‘same’, use_bias=False)(self.input_layer)
            conv1: Conv1dConfig::new(1, self.num_channels, 3) // ✅ Kernel size fixed
                .with_padding(PaddingConfig1d::Same) // ✅ Same padding
                .with_bias(false)// ✅ Matches Keras
                .init(device),
            batch_norm1: BatchNormConfig::new(self.num_channels).init(device), // Conv1D → D = 2
            fc1: LinearConfig::new(self.board_x * self.board_y * self.num_channels, 16).init(device),
            batch_norm2: BatchNormConfig::new(16).init(device), // Dense → D = 1
            fc2: LinearConfig::new(16, 16).init(device),
            batch_norm3: BatchNormConfig::new(16).init(device), // Dense → D = 1
            dropout: DropoutConfig::new(self.dropout).init(),
            pi: LinearConfig::new(16, self.action_size).init(device), // Policy head
            v: LinearConfig::new(16, 1).init(device),        // Value head
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.conv1.forward(x.float());
        let x = self.batch_norm1.forward(x);
        let x = Relu::new().forward(x); // ✅ Explicit ReLU activation

        let x = x.flatten(1, 2); // ✅ Correct flatten with start & end dim

        let x = self.fc1.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = Relu::new().forward(x); // ✅ Explicit ReLU activation
        let x = self.dropout.forward(x);

        let x = self.fc2.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = Relu::new().forward(x); // ✅ Explicit ReLU activation
        let x = self.dropout.forward(x);

        let pi = softmax(self.pi.forward(x.clone()), 1); // ✅ Explicit Softmax
        let v = tanh(self.v.forward(x)); // ✅ Explicit Tanh
        
        (pi, v)
    }
    // # Shapes
    //   - Input [batch_size, height, width, 1] (last dim is next mover)
    //   - Output [batch_size, target_pis, 1] (last dim is value)
    // pub fn forward(&self, boards: Tensor<B, 4>) -> Tensor<B, 3> {
    //     let [batch_size, height, width, next_mover] = boards.dims();
    //
    //     // Create a channel at the second dimension.
    //     let x = boards.reshape([batch_size, 1, height, width, next_mover]);
    //
    //
    //     let x = self.conv1.forward(x); // [batch_size, 8, _, _]
    //     let x = self.dropout.forward(x);
    //     let x = self.conv2.forward(x); // [batch_size, 16, _, _]
    //     let x = self.dropout.forward(x);
    //     let x = self.activation.forward(x);
    //
    //     let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
    //     let x = x.reshape([batch_size, 16 * 8 * 8]);
    //     let x = self.linear1.forward(x);
    //     let x = self.dropout.forward(x);
    //     let x = self.activation.forward(x);
    //
    //     self.linear2.forward(x) // [batch_size, num_classes]
    // }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model() {

    }
}