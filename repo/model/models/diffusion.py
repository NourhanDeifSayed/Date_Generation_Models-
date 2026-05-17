import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def make_linear_schedule(T: int = 50,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> Dict[str, np.ndarray]:
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


class NoisePredictor(tf.keras.Model):
    def __init__(self, condition_dim: int = 21,
                 hidden_units=(64, 64),
                 time_emb_dim: int = 16):

        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation="swish"),
        ])

        self.cond_proj = tf.keras.layers.Dense(time_emb_dim)

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="swish"),
            tf.keras.layers.Dense(64, activation="swish"),
            tf.keras.layers.Dense(3),
        ])

    @staticmethod
    def sinusoidal_embedding(t, dim):
        half = dim // 2
        freqs = tf.exp(-np.log(10000.0) * tf.cast(tf.range(half), tf.float32) / half)
        t = tf.cast(t, tf.float32)[:, None]
        return tf.concat([tf.sin(t * freqs), tf.cos(t * freqs)], axis=-1)

    def call(self, x_t, t, cond, training=False):
        t_emb = self.time_proj(self.sinusoidal_embedding(t, self.time_emb_dim))
        c_emb = self.cond_proj(cond)
        x = tf.concat([x_t, t_emb, c_emb], axis=-1)
        return self.net(x, training=training)


class ConditionalDiffusion:
    def __init__(self, config):

        self.T = config.get("T", 50)   # 🔥 reduced

        sched = make_linear_schedule(self.T)

        self.betas = tf.constant(sched["betas"])
        self.alphas = tf.constant(sched["alphas"])
        self.alpha_bars = tf.constant(sched["alpha_bars"])

        self.model = NoisePredictor()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        logger.info(f"Diffusion initialized fast mode T={self.T}")

    @tf.function
    def q_sample(self, x0, t, noise):
        ab = tf.gather(self.alpha_bars, t)[:, None]
        return tf.sqrt(ab) * x0 + tf.sqrt(1 - ab) * noise

    @tf.function
    def train_step(self, x0, cond):
        b = tf.shape(x0)[0]
        t = tf.random.uniform([b], 0, self.T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x0))
        xt = self.q_sample(x0, t, noise)

        with tf.GradientTape() as tape:
            pred = self.model(xt, t, cond)
            loss = tf.reduce_mean(tf.square(noise - pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train_epoch(self, batches):
        losses = []
        for cond, date in batches:
            loss = self.train_step(
                tf.constant(date, tf.float32),
                tf.constant(cond, tf.float32),
            )
            losses.append(float(loss))
        return float(np.mean(losses))

    def generate(self, cond_np):

        cond = tf.constant(cond_np[None, :], tf.float32)
        x = tf.random.normal([1, 3])

        steps = np.linspace(self.T - 1, 0, 25).astype(int)  # 🔥 FAST SAMPLING

        for t in steps:
            tt = tf.constant([t], tf.int32)
            pred = self.model(x, tt, cond)

            alpha = self.alphas[t]
            abar = self.alpha_bars[t]

            coef = (1 - alpha) / tf.sqrt(1 - abar)
            mean = (1 / tf.sqrt(alpha)) * (x - coef * pred)

            x = mean  # no noise for speed

        return np.clip(x.numpy()[0], 0, 1)

    def save(self, path):
        self.model.save_weights(f"{path}/diffusion.weights.h5")

    def load(self, path):
        self.model(tf.zeros([1,3]), tf.zeros([1], tf.int32), tf.zeros([1,21]))
        self.model.load_weights(f"{path}/diffusion.weights.h5")