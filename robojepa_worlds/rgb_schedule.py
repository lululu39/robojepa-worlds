"""Restartable clip-group shuffling; every real frame occurs once per epoch."""
import math
import random


class FrameSchedule:
    def __init__(self, frame_counts, batch_size=4, group_size=4, seed=42):
        if not frame_counts or min(frame_counts) <= 0 or batch_size <= 0 or group_size <= 0:
            raise ValueError('Nonempty clips and positive batch/group sizes required')
        self.counts = list(frame_counts)
        self.batch_size, self.group_size, self.seed = batch_size, group_size, seed
        self.epoch = -1
        self.position = 0
        self.plan = []

    def _plan(self, epoch):
        rng = random.Random(self.seed + epoch * 1000003)
        order = list(range(len(self.counts))); rng.shuffle(order)
        plan = []
        for left in range(0, len(order), self.group_size):
            group = order[left:left+self.group_size]
            frames = {}
            for clip in group:
                frames[clip] = list(range(self.counts[clip])); rng.shuffle(frames[clip])
            for start in range(0, max(self.counts[c] for c in group), self.batch_size):
                plan.append([(c, frames[c][start:start+self.batch_size]) for c in group if start < self.counts[c]])
        return plan

    def at(self, position):
        # Epoch lengths can differ for unequal clip lengths, so advance cumulatively.
        if position < self.position or not self.plan:
            self.epoch, self.position = 0, 0
            self.plan = self._plan(0)
        while position >= self.position + len(self.plan):
            self.position += len(self.plan); self.epoch += 1
            self.plan = self._plan(self.epoch)
        return self.plan[position-self.position]


def lr_scale(step, cfg):
    warmup = cfg['adapter_warmup_steps']
    if step < warmup:
        return min(1., (step+1)/cfg['lr_warmup_steps'])
    progress = (step-warmup)/max(1, cfg['max_steps']-warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., progress)))
