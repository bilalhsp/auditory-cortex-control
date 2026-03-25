import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ml_utils import utils
from audioldm import build_model
from auditory_cortex import pretrained_dir
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def match_dimensions(tensor, shape):
    """Return view of the input tensor to allow broadcasting with the shape."""
    tensor_shape = tensor.shape
    while len(tensor_shape) < len(shape):
        tensor_shape = tensor_shape + (1,) 
    return tensor.view(tensor_shape)

class DiffusionDDPM(nn.Module, ABC):
    """Implements the iDDPM model from 'Improved Denoising Diffusion Probabilistic Models' (Nichol et al. 2021).
    This is a denoising diffusion model that uses the noise prediction loss.
    """
    def __init__(self, beta_min=0.0001, beta_max=0.02, T=1000):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

        self.dtype = torch.float32
        
        self.define_schedule(beta_min, beta_max, self.T)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def define_schedule(self, beta_min, beta_max, n_steps):
        self.betas = np.linspace(beta_min, beta_max, n_steps, dtype=np.float32)
        self.alphas = 1 - self.betas     # alpha = 1 - beta
        self.alpha_bars = np.cumprod(self.alphas)
        self.sigma_t = np.sqrt((1 - self.alpha_bars) / self.alpha_bars)


    @abstractmethod
    def forward(self, x, t, **kwargs):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        pass

    def sigma_schedule(self, num_steps, rho=7, sigma_min=0.0064, sigma_max=80):
        if num_steps == 1:
            return np.array([sigma_max])
        step_indices = np.arange(num_steps)
        return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


    def sample_xt(self, x0, sigma):
        x_sigma = x0 + sigma*torch.randn(x0.shape, dtype=self.dtype, device=x0.device)
        return x_sigma
    
    def estimate_epsilon(self, x, t, **kwargs):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
            kwargs: additional keyword arguments to be passed to the forward method
        """ 
        return self.forward(x, t, **kwargs)

    def iddpm_sigma_schedule(self, num_steps=1000, idx_offset=0):
        """Implements the iDDPM sigma schedule, by using linear steps from max_index-j0 to 0.
        Sigma schedule is defined by the value of sigma at each of these indices.

        EDM paper uses an offset j0 to start the schedule from a higher value of sigma.
        For cosine schedule, they used j0=8, which gives maximum value of sigma=80.
        For linear schedule, this is j0=69 , which gives maximum value of sigma=80.
        
        Args:
            num_steps: number of steps to generate the sample
            idx_offset: offset to start the schedule from, same as j0 in EDM paper.
        """
        step_indices = np.arange(num_steps)
        t_indices =  self.T - 1 - np.floor(idx_offset + (self.T-1-idx_offset)/(num_steps-1)*step_indices + 0.5).astype(int) 
        return self.sigma_t[t_indices]
    
    def get_sigma_index(self, sigma, beta_min=0.1, beta_d=19.9):
        # sigma = torch.as_tensor(sigma).to(self.dtype)
        return ((beta_min ** 2 + 2 * beta_d * (1 + sigma ** 2).log()).sqrt() - beta_min) / beta_d


    def condition_index(self, sigma):
        """Returns the condition index for the given sigma value.
        This is refered to as C_cond(sigma) in the EDM paper.
        DNN in iDDPM is conditioned on index of markov chain (t: 0-T) at which the sample is generated.
        For DDIM, they condition directory on the sigma value
        """
        # return np.argmin(np.abs(self.sigma_t - sigma)) #  # -1 because we start from 0 index
        # This matches DAPS inverse conditioning scheme
        # beta_d          = 19.9         # Extent of the noise level schedule.
        # beta_min        = 0.1          # Initial slope of the noise level schedule.
        # M               = 1000         # Original number of timesteps in the DDPM formulation.
        # M = self.T
        # beta_min = self.beta_min
        # beta_d = self.beta_max - self.beta_min
        # c_noise = (M - 1) * self.get_sigma_index(sigma, beta_min, beta_d)

        # Trying out index of closest sigma value from the precomputed sigma_t
        sigma_t = torch.tensor(self.sigma_t).to(sigma.device)
        c_noise = torch.argmin(torch.abs(sigma_t[:,None] - sigma[None,:]), dim=0)
        return c_noise
    
    def score_estimate(self, xt, sigma, **kwargs):
        """Estimate the score at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((xt.shape[0],), sigma, dtype=self.dtype, device=xt.device)
        c_noise = self.condition_index(sigma)
        c_in = 1 / torch.sqrt(sigma**2 + 1)

        c_in = match_dimensions(c_in, xt.shape)
        eps = self.estimate_epsilon(c_in*xt, c_noise, **kwargs)
        denom = match_dimensions(sigma, xt.shape)
        return - eps / denom


    def denoised_estimate(self, x_hat, sigma, **kwargs):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level float or tensor of shape (N,)

        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x_hat.shape[0],), sigma, dtype=self.dtype, device=x_hat.device)
        c_noise = self.condition_index(sigma)
        # c_noise = torch.full((x_hat.shape[0],), c_noise, dtype=self.dtype, device=x_hat.device)
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_out = -sigma

        c_in = match_dimensions(c_in, x_hat.shape)
        c_out = match_dimensions(c_out, x_hat.shape)
        eps = self.estimate_epsilon(c_in*x_hat, c_noise, **kwargs)
        return x_hat + c_out*eps
    
    def denoised_estimate_with_score(self, x_hat, sigma, **kwargs):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level float or tensor of shape (N,)

        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x_hat.shape[0],), sigma, dtype=self.dtype, device=x_hat.device)
        c_noise = self.condition_index(sigma)
        # c_noise = torch.full((x_hat.shape[0],), c_noise, dtype=self.dtype, device=x_hat.device)
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_out = -sigma

        c_in = match_dimensions(c_in, x_hat.shape)
        c_out = match_dimensions(c_out, x_hat.shape)
        eps = self.estimate_epsilon(c_in*x_hat, c_noise, **kwargs)

        return x_hat + c_out*eps, - eps / match_dimensions(sigma, x_hat.shape) 
    
    def dx_estimate(self, x, sigma, **kwargs):
        """Returns the dx estimate D_{\theta}(x, sigmal) as given in EDM paper.
        This is used to compute the reverse diffusion step.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level

        """
        # simplified expression for d_cur, for s(t) = 1, sigma(t) = t
        denoised_x = self.denoised_estimate(x, sigma, **kwargs)
        out = (x - denoised_x)/sigma
        return out
        
    def validate_sigma(self, sigma):
        """Returns the inverse of the function sigma(t)."""
        index = np.argmin(np.abs(self.sigma_t-sigma))
        sigma_true = self.sigma_t[index]
        return sigma_true
    
    def sigma_inv(self, sigma):
        """Returns the inverse of the function sigma(t)."""
        index = np.argmin(np.abs(self.sigma_t-sigma))
        sigma_true = self.sigma_t[index]
        return sigma_true

    def get_sigma_t(self, t):
        """Returns the inverse of the function sigma(t)."""
        return t
    
    @torch.no_grad()
    def ddpm_ode_sample(
        self, n_evals=100, start_x=None, **kwargs
        ):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        shape = kwargs.get("shape", (1, 3, 256, 256))
        start_t = kwargs.get("start_t", self.T)
        offset = self.T - start_t
        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps, offset)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is not None:
            x = start_x.to(self.device, dtype=self.dtype)
        else:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current, **kwargs)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next, **kwargs )
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def edm_ode_sample(
        self, n_evals=100, start_x=None, **kwargs):
        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        sigma_min=kwargs.get("sigma_min", 0.0064)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x.to(self.device, dtype=self.dtype)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current, **kwargs)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next, **kwargs)
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous().to(torch.float32)
        return x
    

    @torch.no_grad()
    def ddpm_sde_sample(
        self, n_evals=100, shape=(1, 3, 256, 256), **kwargs):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat, **kwargs)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next, **kwargs)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x

    @torch.no_grad()
    def edm_sde_sample(
        self, n_evals=100, start_x=None, **kwargs):

        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        sigma_min=kwargs.get("sigma_min", 0.0064)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x.to(self.device, dtype=self.dtype)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat, **kwargs)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next, **kwargs)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def generate(self, method, batch_size=1, **kwargs):
        """Generate samples using the diffusion model as a denoiser.
        Args:
            method: method to use for generation, e.g. 'ddpm', 'sde', etc.
            batch_size: number of samples to generate
            **kwargs: additional arguments for the generation method
        """
        shape = (batch_size, 3, 256, 256)
        if method == 'ddpm_ode':
            return self.ddpm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_ode':
            return self.edm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_sde':
            return self.edm_sde_sample(shape=shape, **kwargs)
        elif method == 'ddpm_sde':
            return self.ddpm_sde_sample(shape=shape, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")

#########################           Implementing DDIM sampler/ as a sanity check #######################

    def get_alpha_bar(self, t, s=None):
        """Get alpha_bar for transition from s to t. 
        t = [1, T], so appropriate adjust indices for zero-based index.
        
        Args:
            t: (N,) tensor of target time step
            s: (N,) tensor of initial, if None then treat as zero.
        """
        t = t - 1  # convert to zero-based index
        alpha_bar_t = torch.from_numpy(self.alpha_bars).to(t.device)[t.to(torch.int64)]
        if s is not None:
            s = s - 1  # convert to zero-based index
            alpha_bar_s = torch.from_numpy(self.alpha_bars).to(s.device)[s.to(torch.int64)]
            alpha_bar_t = alpha_bar_t / alpha_bar_s
        return alpha_bar_t
    
    def get_sigma_t(self, t, s=None):
        """
        Get sigma_t for transition from s to t. If s is None,
        then treat s = t-1 (i.e. sigma_t = sqrt(Beta_t)). 

        Args:
            t: (N,) tensor of target time step
            s: (N,) tensor of initial, if None then treat as zero.
        """
        if s is None:
            s = t - 1  # treat s as t-1 if not provided
        alpha_bar_st = self.get_alpha_bar(t, s)
        sigma_t = torch.sqrt(1 - alpha_bar_st)

        if self.reverse_var == 'beta_tilde':
            alpha_bar_s = self.get_alpha_bar(s)
            alpha_bar_t = self.get_alpha_bar(t)
            sigma_t = sigma_t * torch.sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t))
        return sigma_t
    
    def score_estimate(self, x, t, **kwargs):
        """Estimate the score at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        # eps = self.estimator(x, t)[:,:3]
        eps = self.forward(x, t, **kwargs)
        denom = match_dimensions(torch.sqrt(1 - self.get_alpha_bar(t)), eps.shape)
        return - eps / denom #torch.sqrt(1 - self.get_alpha_bar(t)).view(-1, 1, 1, 1)
    
    def denoising_step(self, x, t, s=None, stoc=True, **kwargs):
        """Denoising step from x_t to x_s. 
        Predict p(x_s | x_t) using Tweedie's formula:

        Args:
            x: (N, C, H, W) tensor of current sample x_t at t.
            t: (N,) tensor of current time step.
            s: (N,) tensor of next time step, if None then treat as zero.
            stoc: whether to add stochastic term in the denoising step
        """
        
        score = self.score_estimate(x, t, **kwargs)
        alpha_st = self.get_alpha_bar(t, s)
        alpha_st = match_dimensions(alpha_st, x.shape)
        # tweedie's formula: for y = x + sqrt(1-alpha)*z, the denoised value is;
        x = x + (1 - alpha_st)*score
        # Normalize by alpha_bar to get the correct scale 
        # because noising process scales x by sqrt(alpha_bar)
        # y = torch.sqrt(alpha)*x + torch.sqrt(1 - alpha)*z
        x = x / torch.sqrt(alpha_st)

        if stoc and s is not None: # adds stochastic term
            z = torch.randn_like(x)
            sigma_st = self.get_sigma_t(t, s)
            sigma_st = match_dimensions(sigma_st, x.shape)
            x = x + sigma_st*z
        return x
    
    @torch.no_grad()
    def ddim_sample(
            self, n_evals=100, start_x=None, start_t=None, show_progress=False, 
            **kwargs
            ):
        """Generates samples from the diffusion model using DDIM sampler.
        To sample p(x_s | x_t) it implements Eq. 7 from DDIM paper (Song 2021),
        with variance set to zero. 
        This corresponds to the deterministic sampling process with eta=0. 
        
        Args:
            n_evals: number of steps to generate the sample
            start_x: initial sample to start from, if None then random noise is used
            start_t: initial time step to start from, if None then T is used

        """
        shape = kwargs.get("shape", (1, 3, 256, 256))
        if start_x is None:
            x = torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x
        if start_t is None:
            start_t = self.T
        x = x.to(self.device)
        B = x.shape[0]
        h = start_t/n_evals     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_evals)]
        for i, time_step in enumerate(tqdm.tqdm(timesteps, disable=not show_progress)): 
            if i < (n_evals - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)
            x_0 = self.denoising_step(x, t, s=None, stoc=False, **kwargs)
            if s is None:
                x = x_0
            else:
                alpha_bar_s = self.get_alpha_bar(s)
                alpha_bar_s = match_dimensions(alpha_bar_s, x.shape)
                alpha_bar_t = self.get_alpha_bar(t)
                alpha_bar_t = match_dimensions(alpha_bar_t, x.shape)
                
                x = torch.sqrt(alpha_bar_s)*x_0 + torch.sqrt(1-alpha_bar_s)*(x - torch.sqrt(alpha_bar_t)*x_0)/(torch.sqrt(1-alpha_bar_t))
        x = x.contiguous()
        return x



class AudioLDM(DiffusionDDPM):
    def __init__(self, model_path=None, beta_min=0.0015, beta_max=0.0195, T=1000):
        super().__init__(beta_min, beta_max, T)
        if model_path is None:
            # model_path = '/scratch/gilbreth/ahmedb/audioldm/audioldm-s-full.ckpt'
            model_path = pretrained_dir / 'audioldm/audioldm-s-full.ckpt'
        self.ldm = build_model(ckpt_path=model_path)
        self.ldm = torch.compile(self.ldm)
        self.null_embedding = self.get_null_condition(1)

        # moving unused parts to cpu...testtt
        self.ldm.cond_stage_model.cpu()
        self.ldm.first_stage_model.encoder.cpu()


    def define_schedule(self, beta_min, beta_max, n_steps):
        self.betas = np.linspace(beta_min**0.5, beta_max**0.5, n_steps, dtype=np.float32)**2
        self.alphas = 1 - self.betas     # alpha = 1 - beta
        self.alpha_bars = np.cumprod(self.alphas)
        self.sigma_t = np.sqrt((1 - self.alpha_bars) / self.alpha_bars)
        

    def forward(self, x, t, **kwargs):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
            c: (N, D) tensor of conditioning embeddings
        """
        cond = kwargs.get("c", None)
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        
        if guidance_scale == 1.0:
            e_t = self.ldm.apply_model(x, t, cond)
        else:
            null_cond = self.null_embedding.expand(x.shape[0], -1, -1)
            
            c_in = torch.cat([null_cond, cond], dim=0)
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            e_t_uncond, e_t = self.ldm.apply_model(x, t, c_in).chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        return e_t.to(self.dtype)


    def get_null_condition(self, batch_size):
        """Returns the null text embedding using the CLAP model for a batch of empty strings.
        """
        text_batch = [""] * batch_size
        return self.get_text_embedding(text_batch)

    def get_text_embedding(self, text):
        """Returns the text embedding using the CLAP model for the list of texts.
        """
        assert isinstance(text, list), f"Expected text to be a list, got {type(text)}"
        return self.ldm.cond_stage_model.model.get_text_embedding(
            self.ldm.cond_stage_model.tokenizer(text)
        ).unsqueeze(dim=1)

    def decode_first_stage(self, z):
        """Decodes the latent representation z to a mel spectrogram.
        """
        if(torch.max(torch.abs(z)) > 1e2):
            z = torch.clip(z, min=-10, max=10)
        z = 1.0 / self.ldm.scale_factor * z
        return self.ldm.first_stage_model.decode(z)

    def mel_spectrogram_to_waveform(self, mel):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.ldm.first_stage_model.vocoder(mel)
        # waveform = waveform.cpu().detach().numpy()
        return waveform

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5