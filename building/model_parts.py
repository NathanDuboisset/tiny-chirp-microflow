"""Reusable learnable audio frontends for tiny-chirp models.

- SincnetConv: SincNet-style learnable bandpass filterbank on rank-4 NHWC audio
  (time on the spatial height axis), microflow-compatible after baking.
- GaborConv1D / GaussianPool1D / SquaredModulus / LogCompression: LEAF-style
  learnable Gabor frontend with Gaussian smoothing and log compression.
"""

import math
import numpy as np
import tensorflow as tf


class SincnetConv(tf.keras.layers.Layer):
    """SincNet-style learnable bandpass filterbank operating on rank-4 NHWC audio.

    Input  shape: (batch, time, 1, 1) — time on the spatial height axis.
    Output shape: (batch, T_out, 1, num_filters).
    """

    def __init__(self, num_filters: int, kernel_size: int, stride: int,
                 sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.stride = stride
        self.sample_rate = sample_rate
        # SincNet requires an odd kernel for symmetric filters.
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

    def build(self, input_shape):
        mel_min = 0.0
        mel_max = self._hz_to_mel(self.sample_rate / 2.0)
        mel_points = np.linspace(mel_min, mel_max, self.num_filters + 1)
        hz_points = self._mel_to_hz(mel_points)
        f1_init = hz_points[:-1] / self.sample_rate
        band_init = np.diff(hz_points) / self.sample_rate

        self.f1 = self.add_weight(
            name="f1",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(f1_init),
            trainable=True,
        )
        self.band = self.add_weight(
            name="band",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(band_init),
            trainable=True,
        )
        t = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        self.t = tf.constant(t, dtype=tf.float32)
        window = 0.54 - 0.46 * np.cos(
            2 * math.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )
        self.window = tf.constant(window, dtype=tf.float32)

    def get_filters(self) -> tf.Tensor:
        f1_safe = tf.math.abs(self.f1)
        f2_safe = f1_safe + tf.math.abs(self.band)

        f1_mat = tf.reshape(f1_safe, (1, -1))
        f2_mat = tf.reshape(f2_safe, (1, -1))
        t_mat = tf.reshape(self.t, (-1, 1))

        pi_t = math.pi * t_mat
        # Avoid division by zero at t=0 — center value injected via mask below.
        denom = tf.where(t_mat == 0.0, 1.0, pi_t)
        filters = (
            tf.math.sin(2.0 * math.pi * f2_mat * t_mat)
            - tf.math.sin(2.0 * math.pi * f1_mat * t_mat)
        ) / denom

        # Limit at t=0 is 2(f2 - f1).
        center_values = 2.0 * (f2_mat - f1_mat)
        mask = tf.cast(t_mat == 0.0, tf.float32)
        filters = filters * (1.0 - mask) + center_values * mask

        filters = filters * tf.reshape(self.window, (-1, 1))
        return tf.reshape(filters, (self.kernel_size, 1, self.num_filters))

    def get_filters_nhwc(self) -> tf.Tensor:
        """tf.nn.conv2d filter [k_h, k_w, in_c, out_c]."""
        return tf.reshape(self.get_filters(), (self.kernel_size, 1, 1, self.num_filters))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            inputs,
            self.get_filters_nhwc(),
            strides=[1, self.stride, 1, 1],
            padding="VALID",
            data_format="NHWC",
        )

    def export_to_conv2d(self, name: str = "baked_sinc_conv") -> tf.keras.layers.Conv2D:
        """Bake learned Sinc filters into a static Conv2D for TFLite / microflow."""
        baked = self.get_filters().numpy()
        w = np.reshape(baked, (self.kernel_size, 1, 1, self.num_filters))
        conv_layer = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=(self.kernel_size, 1),
            strides=(self.stride, 1),
            padding="valid",
            use_bias=False,
            name=name,
        )
        conv_layer.build(input_shape=(None, None, 1, 1))
        conv_layer.set_weights([w])
        return conv_layer

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
        })
        return config

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class SincnetConvW(tf.keras.layers.Layer):
    """SincNet on the width (W) axis. Input: (B, H, T, 1) -> (B, H, T_out, num_filters).

    Unlike SincnetConv (kernel on H), kernel_size is honored as-is (no auto-bump
    to odd) so Axon's max filter dim of 16 stays respected. Each H row (chunk)
    is processed independently with the same learnable bandpass filterbank.
    """

    def __init__(self, num_filters: int, kernel_size: int, stride: int,
                 sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate

    def build(self, input_shape):
        mel_min = 0.0
        mel_max = 2595.0 * np.log10(1.0 + (self.sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(mel_min, mel_max, self.num_filters + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        f1_init = hz_points[:-1] / self.sample_rate
        band_init = np.diff(hz_points) / self.sample_rate

        self.f1 = self.add_weight(
            name="f1", shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(f1_init), trainable=True,
        )
        self.band = self.add_weight(
            name="band", shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(band_init), trainable=True,
        )
        # Odd k -> sample lands on t=0; even k -> centered between two taps.
        t = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        self.t = tf.constant(t, dtype=tf.float32)
        window = 0.54 - 0.46 * np.cos(
            2 * math.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )
        self.window = tf.constant(window, dtype=tf.float32)

    def get_filters(self) -> tf.Tensor:
        f1_safe = tf.math.abs(self.f1)
        f2_safe = f1_safe + tf.math.abs(self.band)
        f1_mat = tf.reshape(f1_safe, (1, -1))
        f2_mat = tf.reshape(f2_safe, (1, -1))
        t_mat = tf.reshape(self.t, (-1, 1))
        pi_t = math.pi * t_mat
        denom = tf.where(t_mat == 0.0, 1.0, pi_t)
        filters = (
            tf.math.sin(2.0 * math.pi * f2_mat * t_mat)
            - tf.math.sin(2.0 * math.pi * f1_mat * t_mat)
        ) / denom
        # Only triggered when a sample lands on t=0 (odd kernel). Harmless when even.
        center_values = 2.0 * (f2_mat - f1_mat)
        mask = tf.cast(t_mat == 0.0, tf.float32)
        filters = filters * (1.0 - mask) + center_values * mask
        filters = filters * tf.reshape(self.window, (-1, 1))
        return filters  # (kernel_size, num_filters)

    def get_filters_nhwc(self) -> tf.Tensor:
        # (k_h=1, k_w=kernel_size, in_c=1, out_c=num_filters)
        return tf.reshape(self.get_filters(), (1, self.kernel_size, 1, self.num_filters))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            inputs,
            self.get_filters_nhwc(),
            strides=[1, 1, self.stride, 1],
            padding="VALID",
            data_format="NHWC",
        )

    def export_to_conv2d(self, name: str = "baked_sinc_conv_w") -> tf.keras.layers.Conv2D:
        baked = self.get_filters().numpy()
        w = np.reshape(baked, (1, self.kernel_size, 1, self.num_filters))
        conv_layer = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=(1, self.kernel_size),
            strides=(1, self.stride),
            padding="valid",
            use_bias=False,
            name=name,
        )
        conv_layer.build(input_shape=(None, None, None, 1))
        conv_layer.set_weights([w])
        return conv_layer

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
        })
        return config


class GaborConv1D(tf.keras.layers.Layer):
    """LEAF-style learnable Gabor filterbank (training graph).

    Returns squared modulus of (cos, sin) responses → energy-like features.
    """

    def __init__(self, num_filters: int, kernel_size: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        # Uniform init helps catch high-pitch birds early in training.
        self.center_freqs = self.add_weight(
            shape=(1, 1, num_filters), initializer="random_uniform", name="center_freqs"
        )
        self.bandwidths = self.add_weight(
            shape=(1, 1, num_filters), initializer="ones", name="bandwidths"
        )

    def get_filters(self) -> tf.Tensor:
        limit = (self.kernel_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.kernel_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        env = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        cos_mod = tf.cos(2.0 * math.pi * self.center_freqs * t)
        sin_mod = tf.sin(2.0 * math.pi * self.center_freqs * t)
        return tf.concat([env * cos_mod, env * sin_mod], axis=-1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        conv = tf.nn.conv1d(inputs, self.get_filters(), stride=self.stride, padding="SAME")
        real, imag = tf.split(conv, 2, axis=-1)
        return tf.square(real) + tf.square(imag)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
        })
        return config


class GaussianPool1D(tf.keras.layers.Layer):
    """LEAF-style learnable Gaussian smoothing pool (depthwise)."""

    def __init__(self, num_filters: int, pool_size: int, stride: int, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.stride = stride
        self.bandwidths = self.add_weight(
            shape=(1, num_filters, 1),
            initializer=tf.constant_initializer(0.4),
            name="bandwidths",
        )

    def get_filters(self) -> tf.Tensor:
        limit = (self.pool_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.pool_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        return gauss / tf.reduce_sum(gauss, axis=0, keepdims=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.depthwise_conv2d(
            tf.expand_dims(inputs, axis=1),
            tf.expand_dims(self.get_filters(), axis=0),
            strides=[1, 1, self.stride, 1],
            padding="SAME",
        )[:, 0, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "pool_size": self.pool_size,
            "stride": self.stride,
        })
        return config


class SquaredModulus(tf.keras.layers.Layer):
    """Squared modulus of stacked (cos, sin) channels — used after baking Gabor into Conv1D."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        real, imag = tf.split(inputs, 2, axis=-1)
        return tf.square(real) + tf.square(imag)


class LogCompression(tf.keras.layers.Layer):
    """log(x + epsilon) — TFLite-friendly compression."""

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.log(inputs + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
