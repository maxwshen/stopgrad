import torch

import torch.nn as nn

from models.ema import init_ema, update_ema_net, sync_ema_net


class Lagrangian(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(Lagrangian, self).__init__()
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

    def forward_with_loss(self, x, aug_cond, noise, t, u, step_for_loss_fn):
        warming_up = (
            self.args.do_velocity_warmup
            and step_for_loss_fn < self.args.velocity_warmup_steps
        )

        if warming_up:
            return self.velocity_loss(x, aug_cond, noise, t)
        return self.offdiag_loss(x, aug_cond, noise, t, u)

    def velocity_loss(self, x, aug_cond, noise, t):
        """Flow-matching loss used only during velocity warmup."""
        with torch.amp.autocast("cuda", enabled=False):
            zt = self.wide(1 - t) * noise + self.wide(t) * x
            zt_dot = x - noise
            vt = self.net(t, t, zt, aug_cond)
            return self.squared_norm(vt - zt_dot).mean()

    def offdiag_loss(self, x, aug_cond, noise, t, u):
        """Full Lagrangian objective, with no training-step-dependent branch."""
        return self.lagrangian_loss(x, aug_cond, noise, t, u)

    def lagrangian_loss(self, x, aug_cond, noise, t, u):

        z0 = noise
        z1 = x
        zt = self.wide(1-t) * z0 + self.wide(t) * z1
        variant = self.args.stopgrad_type

        def ftilde_func(t, u, x):
            return self.net(t, u, x, aug_cond)

        def f_func(t, u, x):
            return x + self.wide(u - t) * ftilde_func(t, u, x)

        with torch.amp.autocast("cuda", enabled=False):

            zeros_t = torch.zeros_like(t)
            ones_u = torch.ones_like(u)
            zeros_zt = torch.zeros_like(zt)

            if variant == 'simp':
                # Expand ∂_u f = f̃ + (u-t)∂_u f̃
                # SG the JVP correction and the right-side target,
                # so gradient only flows through f̃_θ(t,u,X_t)
                ftilde, dtilde_du = torch.func.jvp(
                    ftilde_func, (t, u, zt), (zeros_t, ones_u, zeros_zt)
                )
                left = ftilde + self.wide(u - t) * dtilde_du.detach()
                f_tu = zt + self.wide(u - t) * ftilde.detach()
                right = ftilde_func(u, u, f_tu).detach()
                flowmap_loss = self.squared_norm(left - right)

            else:
                # Work with f directly: JVP gives f(t,u,xt) and ∂_u f(t,u,xt)
                f_tu, partial_u = torch.func.jvp(
                    f_func, (t, u, zt), (zeros_t, ones_u, zeros_zt)
                )
                # Materialize lazy ZeroTensors from zero tangent directions
                # so torch.compile's inductor doesn't hit immutable buffers
                f_tu = f_tu.clone()
                partial_u = partial_u.clone()

                if variant == 'orig':
                    # SG the entire right term (velocity target)
                    vu = ftilde_func(u, u, f_tu.detach())
                    flowmap_loss = self.squared_norm(partial_u - vu.detach())
                else:
                    # 'none': full Lagrangian, no stop-gradients
                    vu = ftilde_func(u, u, f_tu)
                    flowmap_loss = self.squared_norm(partial_u - vu)

            if self.args.adaptive_weight_flowmap:
                adp_wt = (flowmap_loss.detach() + self.args.norm_eps) ** self.args.norm_p
                flowmap_loss = flowmap_loss / adp_wt


            # Velocity loss: flow matching at time u with fresh noise
            fresh_noise = torch.randn_like(z1)
            zu_fresh = self.wide(1-u) * fresh_noise + self.wide(u) * z1
            zu_dot_fresh = z1 - fresh_noise
            vu_fresh = ftilde_func(u, u, zu_fresh)
            velocity_loss = self.squared_norm(vu_fresh - zu_dot_fresh)

            loss = (flowmap_loss + velocity_loss).mean()

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

        def ftilde_func(t, u, x):
            return net(t, u, x, aug_cond=None)

        def f_func(t, u, x):
            return x + self.wide(u-t) * ftilde_func(t, u, x)

        time_pairs = self.get_time_pairs(K=nfe, reverse=False)

        noise = torch.randn(samples_shape, dtype=torch.float32, device=device)
        xhat = noise
        ones = torch.ones(xhat.shape[0],).type_as(xhat)
        for i, time_pair in enumerate(time_pairs):
            t, u = time_pair
            assert t < u
            xhat = f_func(t*ones, u*ones, xhat)
        return xhat
