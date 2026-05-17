import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

VOCAB_SIZE = 32
BOS_TOKEN = 31


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(max_len, d_model),
            initializer="random_normal"
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[:seq_len]


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        out = self.mha(
            x, x,
            attention_mask=tf.cast(mask == 0, tf.bool),
            training=training
        )
        return self.norm(x + out)


class FFN(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(dff, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.dense2(self.dense1(x))
        return self.norm(x + out)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dff: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = FFN(d_model, dff)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.attn(x, training=training)
        x = self.drop(x, training=training)
        return self.ffn(x, training=training)


class DateTransformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dff: int = 256,
        max_seq_len: int = 16,
        condition_dim: int = 21,
        dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.cond_proj = tf.keras.layers.Dense(d_model)
        self.pos_enc = LearnedPositionalEncoding(max_seq_len + 1, d_model)

        self.blocks = [
            TransformerBlock(d_model, n_heads, dff, dropout_rate)
            for _ in range(n_layers)
        ]

        self.ln_final = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, tokens: tf.Tensor, condition: tf.Tensor, training: bool = True) -> tf.Tensor:
        cond_emb = tf.expand_dims(self.cond_proj(condition), 1)
        tok_emb = self.tok_emb(tokens)
        x = tf.concat([cond_emb, tok_emb], axis=1)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_final(x)
        return self.head(x)


class ConditionalTransformer:
    SEQ_LEN = 2

    def __init__(self, config: Dict[str, Any]) -> None:
        tf.random.set_seed(config.get("seed", 42))

        self.model = DateTransformer(
            vocab_size=config.get("vocab_size", VOCAB_SIZE),
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            dff=config.get("dff", 256),
            max_seq_len=config.get("max_seq_len", 16),
            condition_dim=config.get("condition_dim", 21),
            dropout_rate=config.get("dropout_rate", 0.1),
        )

        lr = config.get("learning_rate", 1e-4)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, date_tokens: tf.Tensor, conditions: tf.Tensor) -> tf.Tensor:
        B = tf.shape(date_tokens)[0]
        bos = tf.fill([B, 1], BOS_TOKEN)
        inp = tf.concat([bos, date_tokens[:, :1]], axis=1)
        tgt = date_tokens

        with tf.GradientTape() as tape:
            logits = self.model(inp, conditions, training=True)
            loss = self.loss_fn(tgt[:, 0], logits[:, 1]) + self.loss_fn(tgt[:, 1], logits[:, 2])
            loss = loss / 2.0

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train_epoch(self, batches) -> float:
        losses = []
        for cond_batch, date_float_batch in batches:
            day = np.round(date_float_batch[:, 0] * 30).astype(np.int32)
            month = np.round(date_float_batch[:, 1] * 11).astype(np.int32)

            day_tok = np.clip(day, 0, 30)
            month_tok = np.clip(30 + month, 30, 41)

            tokens = np.stack([day_tok, month_tok], axis=1)

            loss = self.train_step(
                tf.constant(tokens, dtype=tf.int32),
                tf.constant(cond_batch, dtype=tf.float32)
            )

            losses.append(float(loss))

        return float(np.mean(losses))

    def generate(self, condition: np.ndarray) -> np.ndarray:
        cond = tf.constant(condition[np.newaxis], dtype=tf.float32)

        generated = tf.constant([[BOS_TOKEN]], dtype=tf.int32)

        logits = self.model(generated, cond, training=False)
        day_tok = int(tf.argmax(logits[0, 1]).numpy())
        day_tok = int(np.clip(day_tok, 0, 30))

        generated = tf.concat([generated, [[day_tok]]], axis=1)

        logits = self.model(generated, cond, training=False)
        month_tok = int(tf.argmax(logits[0, 2]).numpy())
        month_tok = int(np.clip(month_tok, 30, 41))

        day_norm = day_tok / 30.0
        month_norm = (month_tok - 30) / 11.0
        decade_norm = float(condition[20])
        year_norm = decade_norm

        return np.array([day_norm, month_norm, year_norm], dtype=np.float32)

    def save(self, path: str) -> None:
        self.model.save_weights(f"{path}/transformer.weights.h5")

    def load(self, path: str) -> None:
        dummy_tok = tf.zeros([1, 1], dtype=tf.int32)
        dummy_cond = tf.zeros([1, 21], dtype=tf.float32)
        self.model(dummy_tok, dummy_cond, training=False)
        self.model.load_weights(f"{path}/transformer.weights.h5")