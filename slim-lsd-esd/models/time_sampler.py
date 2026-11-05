import torch

def sample_one_timestep(args, batch_size, device):
    low = args.time_min_training
    high = args.time_max_training
    t = torch.distributions.Uniform(low, high).sample((batch_size,)).to(device)
    return t

def sample_two_timesteps(args, batch_size: int, device: torch.device):
    t, u = triangle_sampler(args, batch_size, device, prob_t_eq_u=args.prob_t_eq_u)
    t = t.clamp(min = args.time_min_training, max = args.time_max_training)
    u = u.clamp(min = args.time_min_training, max = args.time_max_training)
    return t, u

def triangle_sampler(args, batch_size, device, prob_t_eq_u=None):

    prob_same = float(prob_t_eq_u if prob_t_eq_u is not None else args.prob_t_eq_u)
    eps = 1e-8

    tmin = torch.as_tensor(args.time_min_training, device=device, dtype=torch.float32)
    tmax = torch.as_tensor(args.time_max_training, device=device, dtype=torch.float32)

    # --- SAME samples: t ~ Unif[tmin, tmax], u=t
    t_same = torch.rand(batch_size, device=device) * (tmax - tmin) + tmin
    u_same = t_same

    # Uniform over triangle {tmin <= t <= u <= tmax}
    U = torch.rand(batch_size, device=device)
    V = torch.rand(batch_size, device=device)

    # t has CDF F(t) = 1 - ((tmax - t)/(tmax - tmin))^2
    t_ns = tmin + (tmax - tmin) * (1.0 - torch.sqrt(U))
    # u conditional uniform on [t, tmax]
    u_ns = t_ns + (tmax - t_ns) * V

    # Ensure strict ordering if you ever need it (optional)
    u_ns = torch.maximum(u_ns, t_ns + eps)
    u_ns = torch.minimum(u_ns, tmax)

    # --- Mix SAME vs NOT-SAME elementwise
    same_mask = torch.rand(batch_size, device=device) < prob_same
    t = torch.where(same_mask, t_same, t_ns)
    u = torch.where(same_mask, u_same, u_ns)

    return t, u



