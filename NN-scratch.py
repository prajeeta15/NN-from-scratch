import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ---------------- Activation Functions ----------------


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# ---------------- Loss and Accuracy ----------------


def cross_entropy(preds, labels):
    m = labels.shape[0]
    log_likelihood = -np.log(preds[range(m), labels])
    loss = np.sum(log_likelihood) / m
    return loss


def compute_accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == labels)

# ---------------- Neural Network ----------------


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5, l2_lambda=0.001):
        self.weights = []
        self.biases = []
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.m = []
        self.v = []
        self.t = 1

        # Layer initialization
        layers = [input_size] + hidden_layers + [output_size]
        for i in range(len(layers) - 1):
            w = 0.01 * np.random.randn(layers[i], layers[i+1])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.m.append(np.zeros_like(w))
            self.v.append(np.zeros_like(w))

    def forward(self, X, training=True):
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []

        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = relu(Z)
            if training:
                mask = (np.random.rand(*A.shape) >
                        self.dropout_rate).astype(float) / (1 - self.dropout_rate)
                A *= mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(np.ones_like(A))
            self.activations.append(A)

        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        A = softmax(Z)
        self.activations.append(A)
        return A

    def backward(self, y_true, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grads_w = []
        grads_b = []
        m = y_true.shape[0]
        y_pred = self.activations[-1]
        delta = y_pred
        delta[range(m), y_true] -= 1
        delta /= m

        # Output layer gradients
        dw = self.activations[-2].T @ delta + self.l2_lambda * self.weights[-1]
        db = np.sum(delta, axis=0, keepdims=True)
        grads_w.insert(0, dw)
        grads_b.insert(0, db)

        # Backpropagation through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = delta @ self.weights[i+1].T * \
                relu_derivative(self.z_values[i])
            delta *= self.dropout_masks[i]
            dw = self.activations[i].T @ delta + \
                self.l2_lambda * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        # Adam update
        for i in range(len(self.weights)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads_w[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grads_w[i] ** 2)
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            self.weights[i] -= learning_rate * \
                m_hat / (np.sqrt(v_hat) + epsilon)
            self.biases[i] -= learning_rate * grads_b[i]
        self.t += 1

# ---------------- Data Loader ----------------


def load_mnist():
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Activation Visualization ----------------


def visualize_activations(model, sample_image, layer_index=0):
    _ = model.forward(sample_image[None, :], training=False)
    activations = model.activations[layer_index + 1]
    plt.imshow(activations.reshape(
        activations.shape[1], -1), aspect='auto', cmap='plasma')
    plt.title(f"Activations at Layer {layer_index + 1}")
    plt.colorbar()
    plt.show()


def show_predictions(model, X_test, y_test, num_samples=10):
    preds = model.forward(X_test, training=False)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {pred_labels[i]}\nTrue: {y_test[i]}")
        plt.axis('off')
    plt.suptitle("MNIST Predictions (model output)")
    plt.tight_layout()
    plt.show()

# ---------------- Training Loop ----------------


def train_model():
    X_train, X_test, y_train, y_test = load_mnist()
    model = NeuralNetwork(input_size=784, hidden_layers=[256, 128, 64, 32], output_size=10,
                          dropout_rate=0.5, l2_lambda=0.001)

    epochs = 10
    batch_size = 64
    lr = 0.001
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            model.forward(X_batch, training=True)
            model.backward(y_batch, learning_rate=lr)

        # Training metrics
        train_preds = model.forward(X_train, training=False)
        loss = cross_entropy(train_preds, y_train)
        acc = compute_accuracy(train_preds, y_train)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Final test accuracy
    test_preds = model.forward(X_test, training=False)
    test_acc = compute_accuracy(test_preds, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Plot training loss and accuracy
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Loss and Accuracy")
    plt.legend()
    plt.show()

    # Visualize activations
    print("Visualizing activations for a sample image...")
    visualize_activations(model, X_test[0], layer_index=0)
    show_predictions(model, X_test, y_test, num_samples=10)


if __name__ == "__main__":
    train_model()
