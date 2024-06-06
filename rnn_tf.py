import tensorflow as tf
import numpy as np
import os

os.environ['TF_ENABLE_MLIR_BRIDGE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.disable_eager_execution()



def rnn_cell(module_name, hidden_dim):
    """
    Helper function to create a specific RNN cell based on the module name.
    """
    if module_name == 'gru':
        return tf.keras.layers.GRUCell(hidden_dim)
    elif module_name == 'lstm':
        return tf.keras.layers.LSTMCell(hidden_dim)
    elif module_name == 'rnn':
        return tf.keras.layers.SimpleRNNCell(hidden_dim)
    else:
        raise ValueError("Unsupported RNN cell type: {}".format(module_name))

def rnn_tforcing(data, parameters):
    """
    Trains an RNN model and generates synthetic data with teacher forcing.

    Parameters:
    - data: Multivariate time series data of shape (num_samples, num_timestamps, num_attributes).
    - parameters: Dictionary containing model parameters.
    """
    num_samples, num_timestamps, num_attributes = data.shape
    module_name = parameters['module']
    hidden_dim = num_attributes
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    
    # Build the RNN model
    inputs = tf.keras.Input(shape=(num_timestamps, num_attributes))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.RNN(rnn_cell(module_name, hidden_dim), return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    # Placeholder for synthetic data
    synthetic_data = []

    # Train the model
    for i in range(iterations):
        # Randomly sample batch from data
        indices = np.random.choice(num_samples, batch_size, replace=False)
        batch_X = data[indices]
        
        # Set the targets to be the next time steps of the input sequences
        batch_Y = np.roll(batch_X, shift=-1, axis=1)  # Shift the input sequences by one time step
        batch_Y[:, -1, :] = 0  # Zero out the last time step targets as there's no next step data
        
        # Use true outputs as inputs for teacher forcing
        history = model.fit(batch_X, batch_Y, epochs=1, verbose=0)
        train_loss = history.history['loss'][0]

        if i % 1000 == 0 or i == (iterations-1):
            print(f"Iteration {i}, Training Loss: {train_loss}")
    
    # Generate synthetic data
    for _ in range(num_samples):
        # Start with a real sample from the dataset
        index = np.random.choice(num_samples)
        real_sample = data[index:index+1]  # Shape (1, num_timestamps, num_attributes)
        synthetic_sample = model.predict(real_sample)
        synthetic_data.append(synthetic_sample[0])  # Append the generated sample without the batch dimension

    return np.array(synthetic_data)