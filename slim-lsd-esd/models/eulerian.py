import torch

import torch.nn as nn

from models.ema import init_ema, update_ema_net, sync_ema_net


class Eulerian(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(Eulerian, self).__init__()
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
        """Full Eulerian objective, with no training-step-dependent branch."""
        return self.eulerian_loss(x, aug_cond, noise, t, u)

    def eulerian_loss(self, x, aug_cond, noise, t, u):

        z0 = noise
        z1 = x
        zt = self.wide(1-t) * z0 + self.wide(t) * z1
        variant = self.args.stopgrad_type

        def ftilde_func(t, u, x):
            return self.net(t, u, x, aug_cond)

        def f_func(t, u, x):
            return x + self.wide(u - t) * ftilde_func(t, u, x)

        with torch.amp.autocast("cuda", enabled=False):

            # Velocity: v(t, xt) = f̃(t, t, xt)
            vt = ftilde_func(t, t, zt)

            if variant == 'none':
                # Full ESD: JVP of f(t, u, x) with tangents (1, 0, v)
                # gives ∂_t f + (∂_x f)·v directly (transport residual)
                ones_t = torch.ones_like(t)
                zeros_u = torch.zeros_like(u)
                _, residual = torch.func.jvp(
                    f_func, (t, u, zt), (ones_t, zeros_u, vt)
                )
                flowmap_loss = self.squared_norm(residual)

            elif variant == 'orig':
                # ESD-orig: separate ∂_t f and (∂_x f)·v, detach the second
                ones_t = torch.ones_like(t)
                zeros_u = torch.zeros_like(u)
                zeros_zt = torch.zeros_like(zt)
                # ∂_t f(t, u, xt)
                _, dt_f = torch.func.jvp(
                    f_func, (t, u, zt), (ones_t, zeros_u, zeros_zt)
                )
                # (∂_x f)(t, u, xt) · vt — no grad needed, result is detached
                with torch.no_grad():
                    _, dx_f_v = torch.func.jvp(
                        f_func, (t, u, zt), (torch.zeros_like(t), zeros_u, vt)
                    )
                residual = dt_f + dx_f_v
                flowmap_loss = self.squared_norm(residual)

            elif variant == 'simp':
                # ESD-simp: target is (u-t)·∂_t f̃ + (∂_x f)·v, detached
                # left = f̃(t, u, xt)
                # right = SG{(u-t)·∂_t f̃(t, u, xt) + (∂_x f)(t, u, xt)·vt}
                ones_t = torch.ones_like(t)
                zeros_u = torch.zeros_like(u)
                zeros_zt = torch.zeros_like(zt)
                # JVP of f̃ w.r.t. t: gives (f̃, ∂_t f̃)
                ftilde, dt_ftilde = torch.func.jvp(
                    ftilde_func, (t, u, zt), (ones_t, zeros_u, zeros_zt)
                )
                # JVP of f w.r.t. x in direction vt — no grad needed, result is detached
                with torch.no_grad():
                    _, dx_f_v = torch.func.jvp(
                        f_func, (t, u, zt), (torch.zeros_like(t), zeros_u, vt)
                    )
                target = self.wide(u - t) * dt_ftilde.detach() + dx_f_v
                flowmap_loss = self.squared_norm(ftilde - target.detach())


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
