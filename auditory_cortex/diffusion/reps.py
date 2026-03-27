from auditory_cortex.utils import set_up_logging
set_up_logging()


import tqdm
import torch
import torchvision
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)



SAMPLER_REGISTRY = {}


def register_sampler(name):
    """Register posterior sampler with the given name"""
    def decorater(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError("Sampler already exists!")
        SAMPLER_REGISTRY[name] = cls
        return cls
    return decorater

def get_sampler(name, *args, **kwargs):
    """Returns posterior sampler based on its name"""
    if name in SAMPLER_REGISTRY:
        return SAMPLER_REGISTRY[name](*args, **kwargs)
    raise NotImplemented("Sampler not found!")

class PosteriorSampler(ABC):
    def __init__(self, diff_model, operator, latent, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.diff_model = diff_model
        self.operator = operator  # measurement operator
        # self.noiser = noiser  # noise model, e.g. Gaussian noise

        # freeze the diffusion model parameters
        for param in self.diff_model.parameters():
            param.requires_grad = False
        self.diff_model.eval()

        self.latent=latent

    @torch.no_grad()
    def get_measurement_signal(self, x):
        """Get the measurement signal y = Ax + noise"""
        y = self.operator(x)
        return y
    
    @abstractmethod
    def generate_sample(self, measurement, start_x=None, **kwargs):
        """Generate a sample from the posterior distribution p(x|y)
        using the diffusion model. 
        
        Args:
            measurement (torch.Tensor): The measurement signal y.
            start_x (torch.Tensor, optional): Initial value for sample x.
                If None, it will be set using randn.
            kwargs: Additional arguments for the sampling process.
        
        """
        NotImplementedError


    def compute_cond_loss(self, x_0_hat, measurement, **kwargs):

        difference = self.operator(x_0_hat) - measurement  # to update any internal states of the operator
        norm = torch.linalg.norm(difference)
        return difference, norm
    

@register_sampler("reps")
class Restart(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False, device=None):
        super().__init__(diff_model, operator, latent, device=device)
        self.name = 'restart-daps'
        self.dtype = diff_model.dtype

    def generate_sample(self, measurement, **kwargs):
        # standard DAPS..
        n_restarts = kwargs.pop('n_restarts', 100)
        sigma_max=kwargs.pop("sigma_max", 80)
        sigma_min=kwargs.pop("sigma_min", 0.1)
        rho = kwargs.pop("rho", 15)

        ode_rho = kwargs.pop("ode_rho", 7)
        sigma_restart = kwargs.pop("sigma_restart", sigma_max)

        n_ode_steps = kwargs.pop("n_ode_steps", 10)
        ode_sigma_min=kwargs.pop("ode_sigma_min", 0.01)

        start_x = kwargs.pop('start_x', None)
        show_progress = kwargs.pop("show_progress", True)
        B = measurement.shape[0] if measurement is not None else 1
        shape = kwargs.pop("shape", (B, 3, 256, 256))
        record = kwargs.pop("record", False)
       
        # map sigma values to valid discrete sigma values...(for iDDPM)
        sigma_min = self.diff_model.validate_sigma(sigma_min)
        sigma_max = self.diff_model.validate_sigma(sigma_max)
        ode_sigma_min = self.diff_model.validate_sigma(ode_sigma_min)
        sigma_restart = self.diff_model.validate_sigma(sigma_restart)

        restart_schedule = self.diff_model.sigma_schedule(n_restarts, rho=rho, sigma_min=sigma_min, sigma_max=sigma_restart)
        if sigma_restart != sigma_max:
            restart_schedule = np.concatenate(([sigma_max], restart_schedule))

        if start_x is None:
            xt = sigma_max*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            print(f"starting from initlization...")
            xt = start_x.to(self.device, dtype=self.dtype)
        print(xt.shape)
        for step in tqdm.tqdm(range(n_restarts), desc="Restarting...", disable=not show_progress):

            sigma_current = restart_schedule[step]
            xt = self.solve_ode(
                n_ode_steps, xt, sigma_max=sigma_current, sigma_min=ode_sigma_min, ode_rho=ode_rho,
                measurement=measurement, **kwargs
                )

            if step != n_restarts-1:
                sigma_next = restart_schedule[step+1]
                sigma_next = self.diff_model.validate_sigma(sigma_next)     # (for iDDPM) get valid discrete sigma values
                # xt = xt + torch.randn_like(xt)*sigma_next
                xt = xt + torch.randn_like(xt)*(sigma_next**2 - ode_sigma_min**2)**0.5

        xt = xt.contiguous().to(torch.float32)
        if self.latent:
            xt = self.decode(xt)
        outputs = (xt, )
        return outputs
    
    def solve_ode(self, num_steps, start_x=None, **kwargs):
        ode_rho = kwargs.pop("ode_rho", 7)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.001)

        sigma_values = self.diff_model.sigma_schedule(num_steps, rho=ode_rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))
        
        x = start_x.to(self.device)
        for step in tqdm.tqdm(range(num_steps), disable=True):
            
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            # (for iDDPM) get valid discrete sigma values
            sigma_current = self.diff_model.validate_sigma(sigma_current)
            sigma_next = self.diff_model.validate_sigma(sigma_next)

            x = self.ode_step(x, sigma_current, sigma_next, step_ratio=step/num_steps, **kwargs)
        x = x.contiguous().to(torch.float32)
        return x

    def ode_step(self, xt, sigma_current, sigma_next, measurement=None, **kwargs):
        intial_lr = kwargs.get("lr", 1.e-4)
        lr_min_ratio = kwargs.get("lr_min_ratio", None)      # minimum learning rate
        tol = kwargs.get("tol", 1e-10)
        num_iters = kwargs.get("num_iters", 50)
        # lam = kwargs.get("lam", self.operator.sigma**2 / sigma_current**2)
        lam = kwargs.get("lam", None)
        c = kwargs.get('c')

        step_ratio = kwargs.get("step_ratio", None)
        if step_ratio is None or lr_min_ratio is None:
            lr = intial_lr
        else:
            lr = self.get_lr(intial_lr, lr_min_ratio, step_ratio)


        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()

        # DDIM ODE solver step...
        x0_hat = self.diff_model.denoised_estimate(xt, sigma_current, c=c).to(torch.float32)
        # end.record()
        # torch.cuda.synchronize()
        # print("Diffusion step ms:", start.elapsed_time(end))


        # if measurement is not None:
        if getattr(self.operator, 'has_svd', False):
            x0_hat = self.closed_form_solution(measurement, x0_hat, sigma_current)
        else:
            x0_hat = self.optimize_image_adam(
                x0_hat, measurement, x0_hat, lam=lam, lr=lr, num_iters=num_iters, 
                tol=tol, verbose=False
                )
        xt = x0_hat + sigma_next*(xt - x0_hat)/ sigma_current

        return xt
    
    def get_lr(self, lr, lr_min_ratio, ratio, rho=1):
        """Calculates the learning rate based on the ratio.
        Args:
            lr (float): Initial learning rate.
            lr_min_ratio (float): Minimum learning rate ratio.
            ratio (float): Ratio to adjust the learning rate. 
                For polynomial decay, ratio is increasing i.e. idx/num_steps.,
                for rate in terms of variances, ratio is the ratio of sigma_t/sigma_max.
        """
        # ratio is between 0 and 1
        multiplier = ((1-ratio)*1**(1/rho) + ratio*lr_min_ratio*1**(1/rho))**rho
        return lr * multiplier

    def regularized_least_squares_loss(self, x, y, x0, lam= 0.1) -> torch.Tensor:

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        if self.latent:
            decoded = self.decode(x)

            data_loss = self.operator.compute_loss(decoded, y)

            # start.record()
            # pred = self.operator(decoded)
            # end.record()
            # torch.cuda.synchronize()
            # print("Deepspeech2 ms:", start.elapsed_time(end))
            

            # data_loss = torch.mean((pred - y) ** 2)
        else:
            # data_loss = torch.mean((self.operator(x) - y) ** 2)
            data_loss = self.operator.compute_loss(x, y)
        reg_loss = lam * torch.mean((x - x0) ** 2)
        return data_loss + reg_loss
    
    def closed_form_solution(self, measurement, x0_hat, sigma_t):

        sr = self.operator.H

        alpha_t = 1
        sigma_y = self.operator.sigma
        A_T_y = sr.Ht(measurement).view(x0_hat.shape)
        b = sigma_y**2 * x0_hat + (sigma_t**2 / alpha_t**2) * A_T_y

        s = sr.singulars()  # flattened singular values
        b_v = sr.Vt(b)
        scale = 1.0 / (alpha_t**2 * sigma_y**2 + sigma_t**2 * s**2)
        x = sr.V(sr.add_zeros(scale * b_v[:, : scale.shape[0]]))
        return x.detach().view(x0_hat.shape)

    @torch.enable_grad()
    def optimize_image_adam(
        self,
        x_init,
        y,
        x0_hat,
        lam=0.1,
        lr=0.01,
        num_iters=50,
        tol=None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Optimize an image tensor x to match target y using Adam with regularized least squares loss.

        Args:
            x_init: Initial image tensor of shape (B, C, H, W)
            y: Target image tensor of same shape as x_init
            x0_hat: Reference image tensor for regularization
            lam: Regularization weight
            lr: Learning rate
            num_iters: Number of optimization steps
            device: Device to run optimization
            verbose: Whether to print loss during optimization

        Returns:
            Optimized image tensor (same shape as x_init)
        """
        device = self.device
        x = x_init.clone().detach().to(device)
        if y is not None:
            y = y.to(device)
        x0_hat = x0_hat.to(device)

        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.regularized_least_squares_loss(x, y, x0_hat, lam=lam)
            loss.backward()
            optimizer.step()

            # if tol is not None and loss.item() < tol:
            #     if verbose:
            #         print(f"Converged at iteration {i+1} with loss {loss.item():.6f}")
            #     break

            # if verbose and (i % max(1, num_iters // 10) == 0):
            #     print(f"Iter {i+1}/{num_iters}, Loss: {loss.item():.6f}")

        return x.detach()
    
    def decode(self, x):
        """Decode latent variable to get pixels and then vocoder to get waveform."""
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        mel = self.diff_model.decode_first_stage(x)
        # end.record()
        # torch.cuda.synchronize()
        # print("VAE decoder ms:", start.elapsed_time(end))

        # start.record()
        wav = self.diff_model.mel_spectrogram_to_waveform(mel)
        # end.record()
        # torch.cuda.synchronize()
        # print("Vocoder ms:", start.elapsed_time(end))
        return wav.squeeze(dim=1)
    

@register_sampler("daps")
class DAPS(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'daps'
        self.dtype = diff_model.dtype

    def generate_sample(self, measurement, **kwargs):
        n_anneal_steps = kwargs.pop('n_anneal_steps', 100)
        n_ode_steps=kwargs.pop("n_ode_steps", 10)
        sigma_max=kwargs.pop("sigma_max", 80)
        sigma_min=kwargs.pop("sigma_min", 0.1)
        ode_sigma_min=kwargs.pop("ode_sigma_min", 0.001)
        rho = kwargs.pop("rho", 7)

        # langevin parameters
        n_lang_steps=kwargs.pop("n_lang_steps", 50)
        lr = kwargs.pop("lr", 1.e-4)
        lr_min_ratio = kwargs.pop("lr_min_ratio", 1.e-2)      # minimum learning rate
        tau = kwargs.pop("tau", 0.01)

        
        start_x = kwargs.pop('start_x', None)
        show_progress = kwargs.pop("show_progress", True)
        B = measurement.shape[0] if measurement is not None else 1
        shape = kwargs.pop("shape", (B, 3, 256, 256))
        record = kwargs.pop("record", False)
       

        # map sigma values to valid discrete sigma values...(for iDDPM)
        sigma_min = self.diff_model.validate_sigma(sigma_min)
        sigma_max = self.diff_model.validate_sigma(sigma_max)
        ode_sigma_min = self.diff_model.validate_sigma(ode_sigma_min)
        # n_anneal_steps = n_anneal_steps + 1  # to include the first step (following DAPS)
        sigma_values = self.diff_model.sigma_schedule(n_anneal_steps + 1, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values[:-1], np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        
        if start_x is None:
            xt = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            xt = start_x.to(self.device, dtype=self.dtype)

        for step in tqdm.tqdm(range(n_anneal_steps), desc="Generating...", disable=not show_progress):
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            sigma_current = self.diff_model.validate_sigma(sigma_current)
            sigma_next = self.diff_model.validate_sigma(sigma_next)

            # 1. ODE solve to get x0_hat
            x0_hat = self.solve_ode(
                num_steps=n_ode_steps, start_x=xt,
                sigma_max=sigma_current, sigma_min=ode_sigma_min, rho=rho,
                measurement=measurement, ratio=step/n_anneal_steps, #needed only for conditioned ODE
                **kwargs   
                )

            # 2. Langevin updates to get x0y
            eta_t = self.get_lr(lr, lr_min_ratio, step/n_anneal_steps)
            x0y, x0hat_loss, x0y_loss = self.langevin_updates(
                x0_hat, measurement, 
                n_lang_steps, eta_t, sigma_current, tau
                )
            
            # 3. Transition to next xt
            if sigma_next == 0:
                xt = x0y
            else:
                xt = x0y + torch.randn_like(x0y) * sigma_next

            if record:
                self._record(xt, x0y, x0_hat, sigma_current, eta_t, x0hat_loss, x0y_loss)
      
        xt = xt.contiguous().to(torch.float32)
        if self.latent:
            xt = self.decode(xt)
        outputs = (xt, )
        return outputs
    
    def get_lr(self, lr, lr_min_ratio, ratio, rho=1):
        """Calculates the learning rate based on the ratio.
        Args:
            lr (float): Initial learning rate.
            lr_min_ratio (float): Minimum learning rate ratio.
            ratio (float): Ratio to adjust the learning rate. 
                For polynomial decay, ratio is increasing i.e. idx/num_steps.,
                for rate in terms of variances, ratio is the ratio of sigma_t/sigma_max.
        """
        # ratio is between 0 and 1
        multiplier = ((1-ratio)*1**(1/rho) + ratio*lr_min_ratio*1**(1/rho))**rho
        return lr * multiplier
    
    def langevin_updates(self, x0_hat, measurement, n_lang_steps, eta_t, sigma, tau=0.01):
        """Performs Langevin updates to condition the sample."""
        x = x0_hat.detach().clone()
        for idx in range(n_lang_steps):
            
            # Langevin score computation
            score, data_loss = self.get_langevin_score(x, x0_hat, measurement, sigma, tau)

            # Langevin update
            x = x + eta_t*score + np.sqrt(2*eta_t)*torch.randn_like(x)

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x), 0, 0 

            if idx ==0:
                x0hat_loss = data_loss
            elif idx == n_lang_steps-1:
                x0y_loss = data_loss
        return x, x0hat_loss, x0y_loss
    
    def get_langevin_score(self, x, x_0_hat, measurement, sigma, tau):
        """Compute the Langevin score for the given parameters."""

        x_tmp = x.clone().detach().requires_grad_(True)
        if self.latent:
            measurement_loss =  ((self.operator(self.decode(x_tmp)) - measurement) ** 2).sum()
        else:
            measurement_loss =  ((self.operator(x_tmp) - measurement) ** 2).sum()
        measurement_grad = torch.autograd.grad(measurement_loss, x_tmp)[0]
        x_tmp.detach_()

        data_term = -measurement_grad/tau**2
        xt_term = (x_0_hat - x)/sigma**2
        return data_term + xt_term, measurement_loss.item()
    
    def solve_ode(self, num_steps, start_x, **kwargs):

        rho = kwargs.pop("rho", 7)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.001)
        c = kwargs.get('c')

        # num_steps = num_steps+1 # to include the first step (following DAPS)
        sigma_values = self.diff_model.sigma_schedule(num_steps+1, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        # sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))
        
        x = start_x.to(self.device)
        for step in tqdm.tqdm(range(num_steps), disable=True):
            
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            sigma_current = self.diff_model.validate_sigma(sigma_current)
            sigma_next = self.diff_model.validate_sigma(sigma_next)

            # get values of time from sigma values
            t_current = self.diff_model.sigma_inv(sigma_current)
            t_next = self.diff_model.sigma_inv(sigma_next)

            x_denoised = self.diff_model.denoised_estimate(x, sigma_current, c=c).to(torch.float32)
            dx = (x - x_denoised)/ sigma_current

            # Euler step...
            x = x + dx * (t_next - t_current)

        x = x.contiguous().to(torch.float32)
        return x
    
    def decode(self, x):
        """Decode latent variable to get pixels and then vocoder to get waveform."""
        mel = self.diff_model.decode_first_stage(x)
        wav = self.diff_model.mel_spectrogram_to_waveform(mel)
        return wav.squeeze(dim=1)
