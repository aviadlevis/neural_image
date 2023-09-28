import jax
import flax
import functools
import optax
from jax import numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from flax.training import train_state, checkpoints

class PixelPredictor(nn.Module):
    scale: float   # Scaling of the input dimension axis (e.g. to make input range [-1,1] or [0,1] along the different dimensions)
    posenc_deg: int = 3
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.sigmoid
    out_channel: int = 1
    do_skip: bool = True
    sigmoid_offset: float = 10.0
    
    def init_params(self, coords, seed=1):
        params = self.init(jax.random.PRNGKey(seed), coords)['params']
        return params.unfreeze() # TODO(pratul): this unfreeze feels sketchy

    def init_state(self, params, num_iters=5000, lr_init=1e-4, lr_final=1e-6, checkpoint_dir=''):
        lr = optax.polynomial_schedule(lr_init, lr_final, 1, num_iters)
        tx = optax.adam(learning_rate=lr)
        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx) 
        # Replicate state for multiple gpus
        state = flax.jax_utils.replicate(state)
        return state
    
    @nn.compact
    def __call__(self, coords):
        pixel_mlp = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        def predict_pixels(coords):
            net_output = pixel_mlp(posenc(coords / self.scale, self.posenc_deg))
            pixels = nn.sigmoid(net_output[..., 0] - self.sigmoid_offset)
            return pixels
        pixels = predict_pixels(coords)
        return pixels
        
class MLP(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
  
    @nn.compact
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)

        return out

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)

def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))
