import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from deepspeech_pytorch.model import DeepSpeech
from transformers import WhisperForConditionalGeneration, AutoProcessor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from wav2letter.models import Wav2LetterRF

# local imports
from auditory_cortex import utils
from auditory_cortex import results_dir, pretrained_dir, cache_dir
from auditory_cortex.neural_data import UCDavisDataset

HF_CACHE_DIR = cache_dir / 'hf_cache'

ENCODING_MODELS_REGISTRY  = {}

def register_encoder(model_name: str):

    def decorator(cls):
        if model_name in ENCODING_MODELS_REGISTRY :
            raise ValueError(f"Encoder {model_name} is already defined!")
        ENCODING_MODELS_REGISTRY[model_name] = cls
        return cls
    return decorator

def create_encoder(model_name, **kwargs):
    if model_name not in ENCODING_MODELS_REGISTRY :
        raise ValueError(f"Encoder {model_name} is not defined!")
    return ENCODING_MODELS_REGISTRY[model_name](model_name=model_name, **kwargs)


class BaseEncoder(torch.nn.Module):
    def __init__(self, 
                model_name='deepspeech2', trf_model=None, layer_id=0, 
                ch_idx=0, task='representation', pad_time=0.35, rms=0.1,
                device=None,
                ):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.rms = rms
        self.model_name = model_name
        self.config = utils.load_dnn_config(model_name=self.model_name)
        self.layers_dict = {         # layer_id: layer_name
            layer['layer_id']:layer['layer_name']
            for layer in self.config['layers']
            }
        
        self.layer_id = layer_id
        self.ch_idx = ch_idx
        self.task = task
        self.trf_model = trf_model

        self.pad_time = pad_time

        

    def _initialize(self):

        self.model.to(self.device)

        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # setting RNN layers to train mode to enable full backward pass
        for m in self.model.modules():
            if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
                m.train()  # enable full backward pass
            else:
                m.eval()   # keep others in eval (e.g., batchnorm, dropout)

        self.register_hooks(self.layer_id)
        self.layer_features = None
        self.ch_idx = self.ch_idx
        self.task = self.task

        self.trf_layer = None
        if self.trf_model is not None:
            self.trf_layer, self.trf_sfreq, self.trf_features, self.trf_delays  = self._get_trf_layer(self.trf_model)

    def set_target(self, task, ch_idx=0):
        self.ch_idx = ch_idx
        self.task = task

    def _get_trf_layer(self, trf_model):
        n_features = trf_model.X_feats_
        targets = trf_model.n_targets_
        n_delays = trf_model._ndelays
        trf_sfreq = trf_model.sfreq

        conv = torch.nn.Conv1d(n_features, targets, n_delays, stride=1, padding=(n_delays-1)).to(self.device)
        weights = torch.tensor(trf_model.coef_).to(dtype=conv.weight.dtype, device=self.device)
        bias = torch.tensor(trf_model.y_mean_).to(dtype=conv.weight.dtype, device=self.device)
        mean = torch.tensor(trf_model.X_mean_).float().reshape(n_features, n_delays).to(self.device)
        std  = torch.tensor(trf_model.X_std_).float().reshape(n_features, n_delays).to(self.device)
        
        with torch.no_grad():
            W = weights / std[:, :, None]
            bias = bias -(W * mean[:, :, None]).sum(dim=(0,1))

            weights_flipped = W.flip(1)
            conv.weight.copy_(weights_flipped.permute(2,0,1))
            conv.bias.copy_(bias.squeeze())

        # 
        conv.eval()
        for param in conv.parameters():
            param.requires_grad = False
        return conv, trf_sfreq, n_features, n_delays

    def register_hooks(self, layer_id):
        layer_name = self.layers_dict[layer_id]
        for name, module in self.model.named_modules():
            if name == layer_name:
                print(f"Registering hook for layer: {layer_name}")
                module.__name__ = layer_name
                module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if 'rnn' in module.__name__:
            features, seq_lengths = nn.utils.rnn.pad_packed_sequence(output[0], batch_first=True)
        else:
            # output = output.squeeze()
            if 'conv' in module.__name__:
                if output.ndim == 4:
                    output = output.reshape(output.shape[0], -1, output.shape[-1])
                output = output.transpose(1,2)
            elif 'coch' in self.model_name:
                if output.ndim > 2:
                    output = output.reshape(output.shape[0]*output.shape[1], -1)
                output = output.transpose(0,1)      # (time, features) format
                if output.shape[1] > 2000:
                    # restrict the output to max 2000 features
                    output = output[:, :2000]   
            features = output
        self.layer_features = features

    def compute_loss(self, x, *args, **kwargs):

        if self.trf_layer is not None and self.task == 'stretch':
            return self.stretch_loss(x)
        elif self.trf_layer is not None and self.task == 'one-hot':
            return self.one_hot_loss(x)
        else:
            return self.repr_matching_loss(x, args[0])

    def repr_matching_loss(self, x, measurement):
        """Loss for matching the outputs to the target."""
        pred = self.forward(x)
        # loss = torch.mean((pred - measurement) ** 2)
        loss = ((pred - measurement) ** 2).flatten(1).mean(-1)      # shape: (B,)
        return loss.sum()   # sum across batch
    
    def stretch_loss(self, x):
        """Loss for activation maximization."""
        # mean_firing_rates
        loss = - self.forward(x)[:, self.ch_idx].flatten(1).mean(-1)      # shape: (B,)
        return loss.sum()   # sum across batch
    
    def one_hot_loss(self, x):
        """Loss for activation maximization."""
        # mean_firing_rates
        one_hot_prediction = F.log_softmax(self.forward(x), dim=1)[:, self.ch_idx]
        loss = - one_hot_prediction.flatten(1).mean(-1)      # shape: (B,)
        return loss.sum()   # sum across batch
    


    def forward(self, x, sampling_rate=16000):
        """Forward pass input and returns layer features."""
        self.layer_features = None

        x = self.rms_normalize(x, target_rms=self.rms)   # normalize the input to a target RMS level
        # pad the waveforms before extracting features...to avoid edge effects in the features...
        x = F.pad(x, (int(self.pad_time*sampling_rate), 0))   
        _ = self.predict(x)

        features = self.layer_features                      # B, T, C
        if self.trf_layer is not None:

            dur = x.shape[-1]/sampling_rate
            size = UCDavisDataset.calculate_num_bins(dur, 1/self.trf_sfreq)

            feats = features.transpose(1,2)                 # B, T, C -> B, C, T
            feats = self.resample_fft(feats, num=size)      # resample to match the TRF sampling frequency
            # feats = self.smooth_interpolate(feats, size=size, sigma=1.0)
            
            pad_samples = int(self.pad_time*self.trf_sfreq + 0.5)
            feats = feats[...,pad_samples:]                         # remove the padded part to align with the original input
            features = self.trf_layer(feats)[...,:-(self.trf_delays-1)]

        return features
    @staticmethod
    def rms_normalize(waveform, target_rms=0.1):
        # works for (time,), (batch, time), or (batch, channels, time)
        squeezed = waveform.ndim == 1
        x = waveform.unsqueeze(0) if squeezed else waveform  # ensure batch dim
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x**2, dim=dims, keepdim=True))
        rms = rms.clamp(min=1e-8)
        out = x * target_rms / rms
        return out.squeeze(0) if squeezed else out

    @staticmethod
    def smooth_interpolate(feats, size, sigma=1.0):
        # apply gaussian blur along time axis first, then interpolate
        kernel_size = int(6 * sigma + 1) | 1          # ensure odd
        x = torch.arange(kernel_size, dtype=feats.dtype, device=feats.device)
        kernel = torch.exp(-0.5 * ((x - kernel_size // 2) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).expand(feats.shape[1], 1, -1)
        
        padded = F.pad(feats, (kernel_size // 2, kernel_size // 2), mode='reflect')
        smoothed = F.conv1d(padded, kernel, groups=feats.shape[1])
        return F.interpolate(smoothed, size=size, mode='linear', align_corners=True)

    @staticmethod
    def resample_fft(x: torch.Tensor, num: int) -> torch.Tensor:
        n = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)

        if num > n:
            # Halve Nyquist bin for even-length input before padding
            if n % 2 == 0:
                X = X.clone()
                X[..., -1] *= 0.5
            padding = torch.zeros(*x.shape[:-1], num // 2 + 1 - X.shape[-1],
                                dtype=X.dtype, device=x.device)
            X_new = torch.cat([X, padding], dim=-1)
        else:
            X_new = X[..., :num // 2 + 1]
            # Halve Nyquist bin for even-length output (it was doubled by truncation)
            if num % 2 == 0:
                X_new = X_new.clone()
                X_new[..., -1] *= 0.5

        return torch.fft.irfft(X_new, n=num, dim=-1) * (num / n)

@register_encoder('deepspeech2')
class DeepSpeechsEncoder(BaseEncoder):
    def __init__(
            self, model_name, 
            trf_model=None, 
            layer_id=0, ch_idx=0, task='representation', 
            pad_time=0.35, device=None
        ):
        super().__init__(
            model_name=model_name, trf_model=trf_model, layer_id=layer_id, 
            ch_idx=ch_idx, task=task, 
            pad_time=pad_time, device=device
        )

        # model specific initialization        
        checkpoint_path = pretrained_dir / model_name / self.config['saved_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only = False)
        self.model = DeepSpeech(**checkpoint['hyper_parameters'])
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)


        self._initialize()

    def predict(self, x):
        """Generate output for the given audio input."""
        spect = self.process_input(x)
        lengths = torch.full((spect.shape[0],), spect.shape[-1], dtype=torch.int64, device=self.device)
        out = self.model(spect, lengths)
        return out

    def process_input(self, y):
        """
        :param y: Audio signal as an array of float numbers
        :return: Spectrogram of the signal
        """
        assert y.ndim < 3, f"expects 1D or 2D inputs, got {y.ndim}-dimensional input"
        sample_rate = self.config.get('sample_rate', 16000)
        window_size = self.config.get('window_size', 0.02)
        window_stride = self.config.get('window_stride', 0.01)
        normalize = self.config.get('normalize', True)

        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)
        # STFT
        D = torch.stft(
            y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=torch.hamming_window(win_length, device=self.device), 
            return_complex=True,
            pad_mode='constant', center=True, normalized=False, onesided=True 
            )
        spect = torch.abs(D)
        spect = torch.log1p(spect)
        if normalize:
            spect = (spect - spect.mean()) / (spect.std() + 1e-5)
        return spect.unsqueeze(1)  # add channel dimension


class WhisperEncoder(BaseEncoder):
    def __init__(
            self, model_name, 
            trf_model=None, 
            layer_id=0, ch_idx=0, task='representation', 
            pad_time=0.35, device=None
        ):
        super().__init__(
            model_name=model_name, trf_model=trf_model, layer_id=layer_id, 
            ch_idx=ch_idx, task=task, 
            pad_time=pad_time, device=device
        )
        self.n_fft: int = 400
        self.n_mels: int = 80
        self.hop_length: int = 160
        self.sampling_rate: int = 16000
        repo_name = self.config['repo_name']
        

        self.model = WhisperForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        
        fb = self.build_mel_filterbank()
        self.register_buffer('mel_filterbank', fb)
        self.register_buffer('window', torch.hann_window(self.n_fft))

        self._initialize()

    def forward(self, x, sampling_rate=16000):
        """Forward pass input and returns layer features."""
        self.layer_features = None
        
        assert sampling_rate == self.sampling_rate, f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}"

        x = self.rms_normalize(x, target_rms=self.rms)   # normalize the input to a target RMS level
        x = F.pad(x, (int(self.pad_time*sampling_rate), 0))   # pad at the beginning to account for TRF delays
        _ = self.predict(x)

        features = self.layer_features 

        # retain only the valid part of the features that corresponds to the input duration (after padding)
        inp_dur = x.shape[-1]/sampling_rate
        fs_feats = features.shape[1]/30             # spect is padded to 30 seconds
        valid_samples = int(inp_dur*fs_feats)
        features = features[:, :valid_samples]                  # B, T, C

        if self.trf_layer is not None:

            size = UCDavisDataset.calculate_num_bins(inp_dur, 1/self.trf_sfreq)

            feats = features.transpose(1,2)                 # B, T, C -> B, C, T
            feats = self.resample_fft(feats, num=size)      # resample to match the TRF sampling frequency

            pad_samples = int(self.pad_time*self.trf_sfreq + 0.5)
            feats = feats[...,pad_samples:]                         # remove the padded part to align with the original input
        
            features = self.trf_layer(feats)[...,:-(self.trf_delays-1)]
            # features = feats.transpose(1,2)

        return features


    def predict(self, x):
        """Generate output for the given audio input."""
        spect = self.process_input(x)
        padded_spect = self.pad_mel_spectrogram(spect)      #← no longer needed

        # padded_spect = self.processor(x.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        # generated_ids = self.model.generate(inputs=padded_spect, max_new_tokens=400)
        # return generated_ids
        # padded_spect = self.pad_mel_spectrogram(spect)  ← no longer needed

        encoder_outputs = self.model.model.encoder(
            input_features=padded_spect,
        )
        return encoder_outputs.last_hidden_state


    def process_input(self, y) -> torch.Tensor:                            # (B, n_mels, T//hop_length)

        stft = torch.stft(y, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_filterbank.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        return (log_spec + 4.0) / 4.0

    def build_mel_filterbank(self):
        n_freqs = 1 + self.n_fft // 2
        min_mel, max_mel = self.hz_to_mel(0.0), self.hz_to_mel(8000.0)
        mel_points = [self.mel_to_hz(min_mel + i * (max_mel - min_mel) / (self.n_mels + 1))
                    for i in range(self.n_mels + 2)]
        fft_freqs = [i * self.sampling_rate / (2 * (n_freqs - 1)) for i in range(n_freqs)]

        fb = torch.zeros(n_freqs, self.n_mels)
        for m in range(self.n_mels):
            fl, fc, fh = mel_points[m], mel_points[m + 1], mel_points[m + 2]
            for k, f in enumerate(fft_freqs):
                if fl < f < fh:
                    fb[k, m] = (f - fl) / (fc - fl) if f <= fc else (fh - f) / (fh - fc)
            fb[:, m] /= (fh - fl)                # Slaney normalisation
        return fb
    
    @staticmethod
    def pad_mel_spectrogram(
        specs: torch.Tensor,                      # (B, n_mels, T')
        target_length: int = 3000,
    ) -> torch.Tensor:                            # (B, n_mels, 3000)
        current_length = specs.shape[-1]
        if current_length < target_length:
            pad_amount = target_length - current_length
            specs = torch.nn.functional.pad(specs, (0, pad_amount))
        else:
            specs = specs[..., :target_length]
        return specs

    @staticmethod
    def hz_to_mel(f):
        min_log_hz = 1000.0
        sp = 200.0 / 3.0
        return (min_log_hz / sp + math.log(f / min_log_hz) / math.log(6.4) * 27.0
                if f >= min_log_hz else f / sp)

    @staticmethod
    def mel_to_hz(m):
        min_log_hz, min_log_mel = 1000.0, 15.0
        sp = 200.0 / 3.0
        return (min_log_hz * math.exp(math.log(6.4) / 27.0 * (m - min_log_mel))
                if m >= min_log_mel else sp * m)


@register_encoder('whisper_tiny')
class WhisperTiny(WhisperEncoder):
    def __init__(self, *args, **krargs):
        super().__init__(*args, **krargs)

@register_encoder('whisper_base')
class WhisperBase(WhisperEncoder):
    def __init__(self, *args, **krargs):
        super().__init__(*args, **krargs)



@register_encoder('wav2vec2')
class Wav2vec2Encoder(BaseEncoder):

    def __init__(
            self, model_name, 
            trf_model=None, 
            layer_id=0, ch_idx=0, task='representation', 
            pad_time=0.35, device=None
        ):
        super().__init__(
            model_name=model_name, trf_model=trf_model, layer_id=layer_id, 
            ch_idx=ch_idx, task=task, 
            pad_time=pad_time, device=device
        )
        repo_name = self.config['repo_name']
        
        self.model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        self.processor = Wav2Vec2Processor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)


        self._initialize()

    def predict(self, x):
        """Generate output for the given audio input."""
        logits = self.model(x).logits
        return logits


@register_encoder('wav2letter_modified')
class Wav2letterEncoder(BaseEncoder):

    def __init__(
            self, model_name, 
            trf_model=None, 
            layer_id=0, ch_idx=0, task='representation', 
            pad_time=0.35, device=None
        ):
        super().__init__(
            model_name=model_name, trf_model=trf_model, layer_id=layer_id, 
            ch_idx=ch_idx, task=task, 
            pad_time=pad_time, device=device
        )

        checkpoint_path = pretrained_dir / model_name / self.config['saved_checkpoint']
        self.model = Wav2LetterRF.load_from_checkpoint(checkpoint_path)

        self._initialize()

    def predict(self, x):
        """Generate output for the given audio input."""
        logits = self.model(x)
        return logits


@register_encoder('speech2text')
class Speech2TextEncoder(BaseEncoder):

    def __init__(
            self, model_name, 
            trf_model=None, 
            layer_id=0, ch_idx=0, task='representation', 
            pad_time=0.35, device=None
        ):
        super().__init__(
            model_name=model_name, trf_model=trf_model, layer_id=layer_id, 
            ch_idx=ch_idx, task=task, 
            pad_time=pad_time, device=device
        )

        self.fft_length = 512
        self.num_mel_bins = 80
        self.sampling_rate = 16000
        self.min_frequency = 20.0
        self.max_frequency = 8000.0

        frame_length_ms = 25.0   # → 400 samples at 16 kHz
        frame_shift_ms = 10.0   # → 160 samples at 16 kHz
        self.frame_length = int(round(frame_length_ms * self.sampling_rate / 1000))
        self.hop_length   = int(round(frame_shift_ms  * self.sampling_rate / 1000))

        self.preemphasis_coeff = 0.97
        self.mel_floor = 1.192092955078125e-07
        
        fb = self.build_mel_filterbank()
        self.register_buffer('mel_filterbank', fb)
        self.register_buffer('window', self.povey_window())


        repo_name = self.config['repo_name']
        self.model = Speech2TextForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        self.processor = Speech2TextProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

        self._initialize()

    def predict(self, x):
        """Generate output for the given audio input."""
        input_features = self.process_input(x)              # (B, T, 80)
        # generated_ids = self.model.generate(input_features, max_new_tokens=200)
        # return generated_ids
        encoder_outputs = self.model.model.encoder(
            input_features=input_features,
        )
        return encoder_outputs.last_hidden_state            # (B, T, C)

    def process_input(self, y) -> torch.Tensor:                   # (B, num_frames, num_mel_bins)
        """
        Differentiable log-mel filterbank spectrogram matching
        HuggingFace Speech2TextFeatureExtractor._extract_fbank_features.

        Kaldi-style: Povey window, per-frame DC removal, preemphasis,
        triangular mel filters in mel space, natural log.
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # --- Framing → (B, num_frames, frame_length) ---
        y = y * (2 ** 15)                          # Kaldi int16 range
        frames   = y.unfold(-1, self.frame_length, self.hop_length)

        # --- DC removal, preemphasis, windowing ---
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = torch.cat([frames[..., :1] * (1 - self.preemphasis_coeff),
                            frames[..., 1:] - self.preemphasis_coeff * frames[..., :-1]], dim=-1)
        frames = frames * self.window

        # --- Power spectrum → mel → log ---
        spec    = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        power   = spec.real ** 2 + spec.imag ** 2                # (B, num_frames, n_freqs)
        log_mel = torch.clamp(power @ self.mel_filterbank, min=self.mel_floor).log()   # (B, num_frames, num_mel_bins)

        # CMVN — matches processor's do_ceptral_normalize=True default
        mean = log_mel.mean(dim=1, keepdim=True)
        std  = log_mel.std(dim=1, keepdim=True).clamp(min=1e-8)
        log_mel = (log_mel - mean) / std

        return log_mel



    def povey_window(self) -> torch.Tensor:
        """
        Kaldi Povey window: a Hann window raised to the power 0.85.
        Non-periodic (symmetric) version, matching kaldi/torchaudio behaviour.
        """
        # symmetric Hann
        n = torch.arange(self.frame_length, dtype=torch.float32)
        hann = 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (self.frame_length - 1))
        return hann.pow(0.85)


    def build_mel_filterbank(self):
        n_freqs = self.fft_length // 2 + 1
        min_mel = self.hz_to_mel_kaldi(self.min_frequency)
        max_mel = self.hz_to_mel_kaldi(self.max_frequency)
        mel_pts = [self.mel_to_hz_kaldi(min_mel + i * (max_mel - min_mel) / (self.num_mel_bins + 1))
                for i in range(self.num_mel_bins + 2)]
        fft_freqs = [i * self.sampling_rate / self.fft_length for i in range(n_freqs)]
        fb = torch.zeros(n_freqs, self.num_mel_bins)
        for m in range(self.num_mel_bins):
            fl, fc, fh = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
            for k, f in enumerate(fft_freqs):
                if fl < f < fh:
                    fb[k, m] = (f - fl) / (fc - fl) if f <= fc else (fh - f) / (fh - fc)
        return fb

    # ---------------------------------------------------------------------------
    # Mel / Hz conversions  (Kaldi / "kaldi" mel scale)
    # ---------------------------------------------------------------------------
    @staticmethod
    def hz_to_mel_kaldi(f: float) -> float:
        """Kaldi mel scale: 1127 * ln(1 + f/700)."""
        return 1127.0 * math.log(1.0 + f / 700.0)
    
    @staticmethod
    def mel_to_hz_kaldi(m: float) -> float:
        return 700.0 * (math.exp(m / 1127.0) - 1.0)


    
        
