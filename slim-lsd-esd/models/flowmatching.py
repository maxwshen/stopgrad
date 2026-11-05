import torch

import torch.nn as nn

from models.ema import init_ema, sync_ema_net, update_ema_net

class FlowMatching(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(FlowMatching, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

    def wide(self, t):
        return t[:, None, None, None]

    def update_ema(self, step):
        if step >= self.args.update_ema_after:
            if self.num_updates == 0:
                # First update: full sync instead of EMA blend
                sync_ema_net(self.net, self.net_ema)
            else:
                update_ema_net(self.net, self.net_ema, self.num_updates, self.args.update_ema_period)
            self.num_updates += 1

    def squared_norm(self, x):
        return x.pow(2).sum(dim=[1,2,3])

    def forward_with_loss(self, x, aug_cond, t, u, noise, step_for_loss_fn):

        step = step_for_loss_fn

        device = x.device
        bsz = x.shape[0]
        z0 = noise
        z1 = x
        zt = self.wide(1-t)* z0 + self.wide(t) * z1
        zt_dot = z1 - z0
 
        def v_func(t, z):
            return self.net(t, t, z, aug_cond)

        with torch.amp.autocast("cuda", enabled=False):
            vt = v_func(t, zt)
            loss = self.squared_norm(vt - zt_dot).mean()
        
        return loss
   

    def sample(self, samples_shape, net=None, device=None, nfe=1):
        net = net if net is not None else self.net_ema

        def v_func(t, z):
            return net(t, t, z, aug_cond=None)

        low = 1e-4
        high = 1 - 1e-4
        
        times = torch.linspace(low, high, nfe)
        if len(times) > 1 :
            dt = times[1] - times[0]
        elif len(times) == 1:
            dt = high - low
        else:
            assert False

        noise = torch.randn(samples_shape, dtype=torch.float32, device=device)
        xhat = noise
        ones = torch.ones(xhat.shape[0],).type_as(xhat)
        for t_scalar in times:
            t_vec = t_scalar * ones
            xhat = xhat + dt * v_func(t_vec, xhat)
        return xhat

