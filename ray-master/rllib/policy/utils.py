import numpy as np
from ray.rllib.utils import try_import_torch
from collections import deque
from skimage.util import view_as_windows

torch, nn = try_import_torch()
import torch.distributions as td
from functools import partial
import itertools

def _make_categorical(x, ncat, shape):
    x = x.reshape((x.shape[0], shape, ncat))
    return td.Categorical(logits=x)

def dist_build(ac_space):
    return partial(_make_categorical, shape=1, ncat=ac_space.n)

def neglogp_actions(pi_logits, actions):
    return nn.functional.cross_entropy(pi_logits, actions, reduction='none')

def sample_actions(logits, device):
    u = torch.rand(logits.shape, dtype=logits.dtype).to(device)
    return torch.argmax(logits - torch.log(-torch.log(u)), dim=1)

def pi_entropy(logits):
    a0 = logits - torch.max(logits, dim=1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), axis=1)

def roll(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def unroll(arr, targetshape):
    s = arr.shape
    return arr.reshape(*targetshape, *s[1:]).swapaxes(0, 1)

def safe_mean(xs):
    return -np.inf if len(xs) == 0 else np.mean(xs)


def pad_and_random_crop(imgs, out, pad):
    """
    Vectorized pad and random crop
    Assumes square images?
    args:
    imgs: shape (B,H,W,C)
    out: output size (e.g. 64)
    """
    # n: batch size.
    imgs = np.pad(imgs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    n = imgs.shape[0]
    img_size = imgs.shape[1] # e.g. 64
    crop_max = img_size - out
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding window
    # combinations of size (out)
    windows = view_as_windows(imgs, (1, out, out, 1))[..., 0,:,:, 0]
    # selects a random window
    # for each batch element
    cropped = windows[np.arange(n), w1, h1]
    cropped = cropped.transpose(0,2,3,1)
    return cropped

def random_cutout_color(imgs, min_cut, max_cut):
    n, h, w, c = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, h, w, c), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        # add random box
        cut_img[h11:h11 + h11, w11:w11 + w11, :] = rand_box[i]
        
        cutouts[i] = cut_img
    return cutouts

def linear_schedule(initial_val, final_val, current_steps, total_steps):
    frac = 1.0 - current_steps / total_steps
    return (initial_val-final_val) * frac + final_val

def horizon_to_gamma(horizon):
    return 1.0 - 1.0/horizon
    
class AdaptiveDiscountTuner:
    def __init__(self, gamma, momentum=0.98, eplenmult=1):
        self.gamma = gamma
        self.momentum = momentum
        self.eplenmult = eplenmult
        
    def update(self, horizon):
        if horizon > 0:
            htarg = horizon * self.eplenmult
            gtarg = horizon_to_gamma(htarg)
            self.gamma = self.gamma * self.momentum + gtarg * (1-self.momentum)
        return self.gamma
    
def flatten01(arr):
    return arr.reshape(-1, *arr.shape[2:])

def flatten012(arr):
    return arr.reshape(-1, *arr.shape[3:])

    
class Returnselector:
    def __init__(self, nenvs, ob_space, ac_space, replay_shape, skips = 0, n_pi = 32, num_return = 5, flat_buffer=False):
        self.skips = skips
        self.n_pi = n_pi
        self.nenvs = nenvs
        
        self.exp_replay = np.empty((*replay_shape, *ob_space.shape), dtype=np.uint8)
        self.vtarg_replay = np.empty(replay_shape, dtype=np.float32)
        
        self.num_return = num_return
        self.ac_space = ac_space
        self.ob_space = ob_space
        
        self.cooldown_counter = skips
        self.replay_index = 0
        self.flat_buffer = flat_buffer

    def update(self, obs_batch, vtarg_batch):
        if self.num_return == 0:
            return False
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        self.exp_replay[self.replay_index] = obs_batch
        self.vtarg_replay[self.replay_index] = vtarg_batch
        
        self.replay_index = (self.replay_index + 1) % self.n_pi
        return self.replay_index == 0
        
    def return_done(self):
        self.cooldown_counter = self.skips
        self.num_return -= 1
        self.replay_index = 0
        
        
    def make_minibatches(self, presleep_pi, num_rollouts):
            if not self.flat_buffer:
                env_segs = list(itertools.product(range(self.n_pi), range(self.nenvs)))
                np.random.shuffle(env_segs)
                env_segs = np.array(env_segs)
                for idx in range(0, len(env_segs), num_rollouts):
                    esinds = env_segs[idx:idx+num_rollouts]
                    mbatch = [flatten01(arr[esinds[:,0], : , esinds[:,1]]) 
                              for arr in (self.exp_replay, self.vtarg_replay, presleep_pi)]
                    yield mbatch
            else:
                nsteps = self.vtarg_replay.shape[1]
                buffsize = self.n_pi * nsteps * self.nenvs
                inds = np.arange(buffsize)
                np.random.shuffle(inds)
                batchsize = num_rollouts * nsteps
                for start in range(0, buffsize, batchsize):
                    end = start+batchsize
                    mbinds = inds[start:end]
                    mbatch = [flatten012(arr)[mbinds] 
                              for arr in (self.exp_replay, self.vtarg_replay, presleep_pi)]
                    
                    yield mbatch
   
        
class RewardNormalizer(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, gamma=0.99, cliprew=10.0, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = gamma
        self.ret_rms = RunningMeanStd(shape=())
        self.cliprew = cliprew
        self.ret = 0. # size updates after first pass
        
    def normalize(self, rews, news, resetrew):
        self.ret = self.ret * self.gamma + rews
        self.ret_rms.update(self.ret)
        rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        if resetrew:
            self.ret[np.array(news, dtype=bool)] = 0. ## Values should be True of False to set positional index
        return rews
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
