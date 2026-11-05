import torch

import torch.nn as nn

from models.ema import init_ema, update_ema_net, sync_ema_net

class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        self.register_buffer("num_updates", torch.tensor(0))
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)


    def update_ema(self, step):
        if step >= self.args.update_ema_after:
            if self.num_updates == 0:
                sync_ema_net(self.net, self.net_ema)
            else:
                update_ema_net(self.net, self.net_ema, self.num_updates, self.args.update_ema_period)
            self.num_updates += 1


    def wide(self, t):
        return t[:, None, None, None]

    def squared_norm(self, x):
        return x.pow(2).sum(dim=[1,2,3])

    def forward_with_loss(self, x, aug_cond, noise, t, u, step_for_loss_fn):

        step = step_for_loss_fn

        device = x.device
        
        e = noise
        
        zt = self.wide(1 - t) * e + self.wide(t) * x

        zt_dot = x - e

        def ftilde_func(t, u, zt):
            return self.net(t, u, zt, aug_cond)

        with torch.amp.autocast("cuda", enabled=False):

            primals = (t, u, zt)

            tangents = (torch.ones_like(t), torch.zeros_like(u), zt_dot)

            pred, jvp = torch.func.jvp(ftilde_func, primals, tangents)

            tgt = (zt_dot + self.wide(u - t) * jvp).detach()
            
            flowmap_loss = self.squared_norm(pred - tgt)

            if self.args.adaptive_weight_flowmap:
                adp_wt = (flowmap_loss.detach() + self.args.norm_eps) ** self.args.norm_p                                                                                                                            
                flowmap_loss = flowmap_loss / adp_wt

            loss = flowmap_loss.mean()  # mean over batch dimension
        
        return loss
    
    def get_time_pairs(self, K, reverse=False):
        time_min = self.args.time_min_sampling
        time_max = self.args.time_max_sampling
        times = torch.linspace(time_min, time_max, K+1)
        pairs = list(zip(times[:-1], times[1:]))
        pairs = [(a.item(),b.item()) for (a,b) in pairs]
        if reverse:
            pairs = list(reversed(pairs))
        return pairs

    def sample(self, samples_shape, net=None, device=None, nfe=1):
        net = net if net is not None else self.net_ema
        time_pairs = self.get_time_pairs(K=nfe, reverse=False)
        noise = torch.randn(samples_shape, dtype=torch.float32, device=device)
        xhat = noise
        ones = torch.ones(xhat.shape[0],).type_as(xhat)
        for i, time_pair in enumerate(time_pairs):
            t, u = time_pair
            assert t < u
            t_vec, u_vec = t*ones, u*ones
            model_out = net(t_vec, u_vec, xhat, aug_cond=None)
            xhat = xhat + self.wide(u_vec-t_vec) * model_out
        return xhat

    def old_sample(self, samples_shape, net=None, device=None):
        assert False
        net = net if net is not None else self.net_ema
        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        t = torch.ones(samples_shape[0], device=device)
        r = torch.zeros(samples_shape[0], device=device)
        u = net(z_1, (t, t - r), aug_cond=None)
        z_0 = z_1 - u
        return z_0


