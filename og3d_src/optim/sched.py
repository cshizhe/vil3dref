"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
import math


def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))

def warmup_cosine(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return 0.5 * (1 + math.cos((step - warmup_step) / (tot_step - warmup_step) * math.pi))

def get_lr_sched(global_step, opts):
    # learning rate scheduling
    if opts.lr_decay == 'linear':
        lr_decay_fn = warmup_linear
    elif opts.lr_decay == 'cosine':
        lr_decay_fn = warmup_cosine

    lr_this_step = opts.learning_rate * lr_decay_fn(
        global_step, opts.warmup_steps, opts.num_train_steps)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step

def get_lr_sched_decay_rate(global_step, opts):
    # learning rate scheduling
    if opts.lr_decay == 'linear':
        lr_decay_fn = warmup_linear
    elif opts.lr_decay == 'cosine':
        lr_decay_fn = warmup_cosine

    lr_decay_rate = lr_decay_fn(
        global_step, opts.warmup_steps, opts.num_train_steps)
    lr_decay_rate = max(lr_decay_rate, 1e-5)
    # if lr_decay_rate <= 0:
    #     lr_decay_rate = 1e-8
    return lr_decay_rate
