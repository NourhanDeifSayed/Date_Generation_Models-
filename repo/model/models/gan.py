import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class Generator(tf.keras.Model):
    def __init__(self, noise_dim: int = 64,
                 condition_dim: int = 21,
                 hidden_units: Tuple[int, ...] = (128, 256, 128)) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        layers = []
        for units in hidden_units:
            layers += [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        layers.append(tf.keras.layers.Dense(3, activation="sigmoid"))
        self.net = tf.keras.Sequential(layers)

    def call(self, noise: tf.Tensor, condition: tf.Tensor,
             training: bool = True) -> tf.Tensor:
        x = tf.concat([noise, condition], axis=-1)
        return self.net(x, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self, condition_dim: int = 21,
                 hidden_units: Tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers = []
        for units in hidden_units:
            layers += [
                tf.keras.layers.Dense(units),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.3),
            ]
        layers.append(tf.keras.layers.Dense(1))
        self.net = tf.keras.Sequential(layers)

    def call(self, date: tf.Tensor, condition: tf.Tensor,
             training: bool = True) -> tf.Tensor:
        x = tf.concat([date, condition], axis=-1)
        return self.net(x, training=training)


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_logits: tf.Tensor,
                       fake_logits: tf.Tensor) -> tf.Tensor:
    real_loss = bce(tf.ones_like(real_logits), real_logits)
    fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def generator_loss(fake_logits: tf.Tensor) -> tf.Tensor:
    return bce(tf.ones_like(fake_logits), fake_logits)


class ConditionalGAN:
    def __init__(self, config: Dict[str, Any]) -> None:
        tf.random.set_seed(config.get("seed", 42))
        np.random.seed(config.get("seed", 42))
        self.noise_dim = config.get("noise_dim", 64)
        cond_dim = config.get("condition_dim", 21)
        self.generator = Generator(
            noise_dim=self.noise_dim,
            condition_dim=cond_dim,
            hidden_units=tuple(config.get("generator_units", [128, 256, 128])),
        )
        self.discriminator = Discriminator(
            condition_dim=cond_dim,
            hidden_units=tuple(config.get("discriminator_units", [128, 64])),
        )
        lr_g = config.get("learning_rate_g", 2e-4)
        lr_d = config.get("learning_rate_d", 2e-4)
        self.opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.5)
        self.opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.5)
        logger.info("ConditionalGAN initialised.")

    @tf.function
    def train_step(self, real_dates: tf.Tensor,
                   conditions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(real_dates)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as d_tape:
            fake_dates = self.generator(noise, conditions, training=True)
            real_logits = self.discriminator(real_dates, conditions, training=True)
            fake_logits = self.discriminator(fake_dates, conditions, training=True)
            d_loss = discriminator_loss(real_logits, fake_logits)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        with tf.GradientTape() as g_tape:
            fake_dates = self.generator(noise, conditions, training=True)
            fake_logits = self.discriminator(fake_dates, conditions, training=True)
            g_loss = generator_loss(fake_logits)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return d_loss, g_loss

    def train_epoch(self, batches) -> Tuple[float, float]:
        d_losses, g_losses = [], []
        for cond_batch, date_batch in batches:
            cond = tf.constant(cond_batch)
            dates = tf.constant(date_batch)
            d_l, g_l = self.train_step(dates, cond)
            d_losses.append(float(d_l))
            g_losses.append(float(g_l))
        return float(np.mean(d_losses)), float(np.mean(g_losses))

    def generate(self, condition: np.ndarray) -> np.ndarray:
        cond = tf.constant(condition[np.newaxis], dtype=tf.float32)
        noise = tf.random.normal([1, self.noise_dim])
        out = self.generator(noise, cond, training=False)
        return out.numpy()[0]

    def save(self, path: str) -> None:
        self.generator.save_weights(f"{path}/gan_generator.weights.h5")
        self.discriminator.save_weights(f"{path}/gan_discriminator.weights.h5")
        logger.info(f"GAN weights saved to '{path}'.")

    def load(self, path: str) -> None:
        dummy_cond: tf.Tensor = tf.zeros([1, 21])
        dummy_noise = tf.zeros([1, self.noise_dim])
        dummy_date = tf.zeros([1, 3])
        self.generator(dummy_noise, dummy_cond, training=False)
        self.discriminator(dummy_date, dummy_cond, training=False)
        self.generator.load_weights(f"{path}/gan_generator.weights.h5")
        self.discriminator.load_weights(f"{path}/gan_discriminator.weights.h5")
        logger.info(f"GAN weights loaded from '{path}'.")