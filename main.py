import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.generator import build_generator
from models.discriminator import build_discriminator

def load_data():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data(path='data/mnist.npz')
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)
    return x_train

def train(generator, discriminator, combined, data, epochs, batch_size):
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_images = data[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs} | D Loss: {0.5 * (d_loss_real + d_loss_fake)} | G Loss: {g_loss}')
            save_images(generator, epoch)

def save_images(generator, epoch):
    noise = np.random.normal(0, 1, (25, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(f"outputs/generated_images/mnist_{epoch}.png")
    plt.close()

def main():
    data = load_data()
    
    generator = build_generator()
    discriminator = build_discriminator()
    
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    z = tf.keras.Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    
    combined = tf.keras.Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    train(generator, discriminator, combined, data, epochs=10000, batch_size=64)

if __name__ == '__main__':
    main()
