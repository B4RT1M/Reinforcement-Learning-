# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:00:26 2020

@author: hongh
"""

from wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

state = env.reset()
print(state.shape)