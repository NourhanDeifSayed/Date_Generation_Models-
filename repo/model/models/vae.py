import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class Encoder(tf.keras.Model):
    def __init__(self, condition_dim: int = 21, latent_dim: int = 32,
                 hidden_units: Tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers = []
        for u in hidden_units:
            layers += [tf.keras.layers.Dense(u, activation="relu")]
        self.backbone = tf.keras.Sequential(layers)
        self.mu_head = tf.keras.layers.Dense(latent_dim)
        self.logvar_head = tf.keras.layers.Dense(latent_dim)

    def call(self, date: tf.Tensor, condition: tf.Tensor,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.concat([date, condition], axis=-1)
        h = self.backbone(x, training=training)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class Decoder(tf.keras.Model):
    def __init__(self, condition_dim: int = 21, latent_dim: int = 32,
                 hidden_units: Tuple[int, ...] = (64, 128)) -> None:
        super().__init__()
        layers = []
        for u in hidden_units:
            layers += [tf.keras.layers.Dense(u, activation="relu")]
        layers.append(tf.keras.layers.Dense(3, activation="sigmoid"))
        self.net = tf.keras.Sequential(layers)

    def call(self, z: tf.Tensor, condition: tf.Tensor,
             training: bool = True) -> tf.Tensor:
        x = tf.concat([z, condition], axis=-1)
        return self.net(x, training=training)


def reparameterise(mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    eps = tf.random.normal(tf.shape(mu))
    return mu + eps * tf.exp(0.5 * logvar)


def vae_loss(real: tf.Tensor, recon: tf.Tensor,
             mu: tf.Tensor, logvar: tf.Tensor,
             kl_weight: float = 0.001) -> tf.Tensor:
    mse = tf.reduce_mean(tf.square(real - recon))
    kl = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return mse + kl_weight * kl, mse, kl


class ConditionalVAE:
    def __init__(self, config: Dict[str, Any]) -> None:
        tf.random.set_seed(config.get("seed", 42))
        np.random.seed(config.get("seed", 42))

        latent_dim = config.get("latent_dim", 32)
        cond_dim = config.get("condition_dim", 21)
        self.kl_weight = config.get("kl_weight", 0.001)

        self.encoder = Encoder(
            condition_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_units=tuple(config.get("encoder_units", [128, 64])),
        )

        self.decoder = Decoder(
            condition_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_units=tuple(config.get("decoder_units", [64, 128])),
        )

        self.latent_dim = latent_dim
        lr = config.get("learning_rate", 1e-3)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(self, real_dates: tf.Tensor,
                   conditions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(real_dates, conditions, training=True)
            z = reparameterise(mu, logvar)
            recon = self.decoder(z, conditions, training=True)
            total, mse, kl = vae_loss(real_dates, recon, mu, logvar, self.kl_weight)

        vars_all = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total, vars_all)
        self.optimizer.apply_gradients(zip(grads, vars_all))
        return total, mse, kl

    def train_epoch(self, batches) -> Tuple[float, float, float]:
        totals, mses, kls = [], [], []
        for cond_batch, date_batch in batches:
            cond = tf.constant(cond_batch)
            dates = tf.constant(date_batch)
            t, m, k = self.train_step(dates, cond)
            totals.append(float(t))
            mses.append(float(m))
            kls.append(float(k))
        return float(np.mean(totals)), float(np.mean(mses)), float(np.mean(kls))

    def generate(self, condition: np.ndarray) -> np.ndarray:
        cond = tf.constant(condition[np.newaxis], dtype=tf.float32)
        z = tf.random.normal([1, self.latent_dim])
        out = self.decoder(z, cond, training=False)
        return out.numpy()[0]

    def save(self, path: str) -> None:
        self.encoder.save_weights(f"{path}/vae_encoder.weights.h5")
        self.decoder.save_weights(f"{path}/vae_decoder.weights.h5")

    def load(self, path: str) -> None:
        dummy_cond = tf.zeros([1, 21])
        dummy_date = tf.zeros([1, 3])
        dummy_z = tf.zeros([1, self.latent_dim])
        self.encoder(dummy_date, dummy_cond, training=False)
        self.decoder(dummy_z, dummy_cond, training=False)
        self.encoder.load_weights(f"{path}/vae_encoder.weights.h5")
        self.decoder.load_weights(f"{path}/vae_decoder.weights.h5")