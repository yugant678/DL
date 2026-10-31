from tensorflow import keras


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build and compile the image classifier used by the Streamlit app."""
    if len(input_shape) != 3 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain three positive dimensions")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
