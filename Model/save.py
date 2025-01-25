def save_model_weights(model, filename="model_weights.h5"):
    """
    Saves the weights of the given model to a file.

    Parameters:
    - model: The trained Keras model whose weights are to be saved.
    - filename: The name of the file to save the weights (default is 'model_weights.h5').

    Returns:
    - None
    """
    model.save_weights(filename)
    print(f"Model weights saved to {filename}")