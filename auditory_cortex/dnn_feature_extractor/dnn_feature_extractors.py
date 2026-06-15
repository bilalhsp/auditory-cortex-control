import os
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from scipy.signal import resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, WhisperForConditionalGeneration, AutoModelForPreTraining
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from transformers import ClapModel, ClapProcessor

# import GPU specific packages...
import deepspeech_pytorch
from deepspeech_pytorch.model import DeepSpeech
import deepspeech_pytorch.loader.data_loader as data_loader
from deepspeech_pytorch.configs.train_config import SpectConfig
from deepspeech_pytorch.checkpoint import CheckpointHandler
import importlib
from wav2letter.models import Wav2LetterRF, Wav2LetterSpect

# local imports
from auditory_cortex import utils
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
from auditory_cortex import results_dir, cache_dir, pretrained_dir

import logging
logger = logging.getLogger(__name__)
import omegaconf

import collections

# _original_load = torch.load

# def patched_load(*args, **kwargs):
#     kwargs["weights_only"] = False
#     return _original_load(*args, **kwargs)

# torch.load = patched_load

# torch.serialization.add_safe_globals([CheckpointHandler])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([tuple])
# torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# torch.serialization.safe_globals([deepspeech_pytorch.configs.train_config.AdamConfig])
# torch.serialization.safe_globals([deepspeech_pytorch.configs.train_config.DeepSpeechConfig])




HF_CACHE_DIR = cache_dir / 'hf_cache'

@register_feature_extractor('wav2letter_modified')
class Wav2LetterModified(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'wav2letter_modified'
        config = utils.load_dnn_config(model_name=self.model_name)
        saved_checkpoint = config['saved_checkpoint']
        checkpoint = os.path.join(pretrained_dir, self.model_name, saved_checkpoint)
        pretrained = config['pretrained']
        if pretrained:		
            model = Wav2LetterRF.load_from_checkpoint(checkpoint)
            logger.info(f"Loading from checkpoint: {checkpoint}")
        else:
            model = Wav2LetterRF()
            logger.info(f"Creating untrained network...!")
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        if not isinstance(aud, torch.Tensor):
            aud = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
            aud = aud.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(aud)
        return out
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        # method not tested yet
        predictions = []
        with torch.no_grad():
            for audio in audio_batch:
                audio = audio.to(self.device)
                predictions.append(label_normalizer(self.model.decode(audio)[0]))# 
        return predictions
    
@register_feature_extractor('wav2vec2')
class Wav2Vec2(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'wav2vec2'

        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        self.processor = Wav2Vec2Processor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)


    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float64)
        input_values = self.processor(input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        with torch.no_grad():
            input_values = input_values.to(self.device)
            logits = self.model(input_values).logits

        return logits
    
    def fwd_pass_tensor(self, aud_tensor):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        logits = self.model(aud_tensor).logits
        return logits

    def transcribe(self, aud):
        """Transcribes speech audio."""
        logits = self.fwd_pass(aud)
        indexes = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(indexes)

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions



@register_feature_extractor('speech2text')
class Speech2Text(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'speech2text'
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Speech2TextForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        self.processor = Speech2TextProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
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
        self.mel_filterbank = fb.to(self.device)
        self.window = self.povey_window().to(self.device)
        # self.register_buffer('mel_filterbank', fb)
        # self.register_buffer('window', self.povey_window())


    def fwd_pass(self, aud):
        # input_features = self.processor(aud,padding=True, sampling_rate=16000, return_tensors="pt").input_features
        if not isinstance(aud, torch.Tensor):
            aud = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
        input_features = self.process_input(aud.unsqueeze(dim=0))

        self.model.eval()
        input_features = input_features.to(self.device)
        generated_ids = self.model.generate(input_features, max_new_tokens=200)
        return generated_ids

    def fwd_pass_tensor(self, aud_spect):
        """
        Forward passes spectrogram of audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud_spect (tensor): spectrogram of input tensor, shape (1, t, 80) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        # feeding decoder the start token...!
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        out = self.model(aud_spect, decoder_input_ids=decoder_input_ids)
        return out
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            # audio, _, target_lens = batch
            for audio in audio_batch:
                input_features = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                pred = label_normalizer(self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
                predictions.append(pred)
                
        return predictions
    
    # def process_input(self, aud):
    #     """Preprocesses the input audio."""
    #     aud = aud.squeeze()
    #     spect = self.processor(
    #         aud, padding=True, sampling_rate=16000, return_tensors="np"
    #         ).input_features[0]
    #     return spect
    
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
    
    
class FeatureExtractorWhisper(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.model_name = model_name
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']    
        model = WhisperForConditionalGeneration.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

        self.n_fft: int = 400
        self.n_mels: int = 80
        self.hop_length: int = 160
        self.sampling_rate: int = 16000

        fb = self.build_mel_filterbank()
        self.mel_filterbank = fb.to(self.device)
        self.window = torch.hann_window(self.n_fft).to(self.device)
        # self.register_buffer('mel_filterbank', fb)
        # self.register_buffer('window', torch.hann_window(self.n_fft))

        


    def process_input(self, aud):
        """Preprocesses the input audio."""
        aud = aud.squeeze()
        spect = self.processor(
            aud, sampling_rate=16000, return_tensors="np", padding="longest"
            ).input_features[0]
        return spect.transpose(1, 0)

    def fwd_pass(self, aud):
        if not isinstance(aud, torch.Tensor):
            aud = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
        spect = self.process_input(aud.unsqueeze(dim=0))
        input_features = self.pad_mel_spectrogram(spect)
        # input_features = self.processor(aud, sampling_rate=16000, return_tensors="pt").input_features
        # with torch.no_grad():
        self.model.eval()
        input_features = input_features.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(inputs=input_features, max_new_tokens=400)
        return generated_ids
    
    def transcribe(self, audio):
        """Transcribes speech audio"""
        predicted_ids = self.fwd_pass(audio)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            for audio in audio_batch:
                input_features = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                predictions.append(self.processor.tokenizer._normalize(transcription))
        return predictions
    
    def predict(self, x):
        """Generate output for the given audio input."""
        spect = self.process_input(x)
        padded_spect = self.pad_mel_spectrogram(spect)
        # padded_spect = self.processor(x.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        generated_ids = self.model.generate(inputs=padded_spect, max_new_tokens=400)
        return generated_ids


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
    

@register_feature_extractor('whisper_tiny')
class WhisperTiny(FeatureExtractorWhisper):
    def __init__(self, shuffled=False):
        model_name = 'whisper_tiny'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('whisper_base')
class WhisperBase(FeatureExtractorWhisper):
    def __init__(self, shuffled=False):
        model_name = 'whisper_base'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('deepspeech2')
class DeepSpeech2(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'deepspeech2'
        config = utils.load_dnn_config(model_name=self.model_name)
        checkpoint_path = os.path.join(pretrained_dir, self.model_name, config['saved_checkpoint'])

        # model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only = False)
        model = DeepSpeech(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['state_dict'])

        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        audio_config = SpectConfig()
        self.parser = data_loader.AudioParser(audio_config, normalize=True)

    
    def process_input(self, aud):
        """Preprocesses the input audio and returns the spectrogram.
        Spectrogram is expected to have features of shape (t, 80).
        
        Args:
            aud (ndarray): single 'wav' input of shape (t,)
        Returns:
            spect (ndarray): spectrogram of the input audio (t, 80)
        
        """
        aud = aud.squeeze()
        spect = self.get_spectrogram(aud).cpu().numpy().transpose(1, 0)
        return spect


    def get_spectrogram(self, aud):
        """Gives spectrogram of audio input."""
        if torch.is_tensor(aud):
            aud = aud.cpu().numpy()
        return self.parser.compute_spectrogram(aud)
    

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

    def fwd_pass(self, aud):

        # test if input is 1 dimensional (audio signal)
        # if aud.ndim == 1:
        #     spect = self.get_spectrogram(aud)
        # spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)

        if not isinstance(aud, torch.Tensor):
            aud = torch.tensor(aud, dtype=torch.float32, device=self.device)#, requires_grad=True)
        spect = self.process_input(aud.unsqueeze(dim=0))

        # length of the spect along time
        lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64, device=self.device)
        spect = spect.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(spect, lengths)
        return out
    
    
    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        # method not tested yet
        predictions = []
        with torch.no_grad():
            self.model.eval()
            for audio in audio_batch:
                spect = self.get_spectrogram(audio.squeeze())
                spect = spect.unsqueeze(dim=0).unsqueeze(dim=0)
                spect = spect.to(self.device)

                # length of the spect along time
                lengths = torch.tensor([spect.shape[-1]], dtype=torch.int64,
                    device=self.device)
                
                out = self.model(spect, lengths)

                output, output_sizes, *_ = out
                decoded_output, _ = self.model.evaluation_decoder.decode(output, output_sizes)
                predictions.append(label_normalizer(decoded_output[0][0]))

        return predictions



@register_feature_extractor('w2v2_audioset')
class W2V2Audioset(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'w2v2_audioset'
        config = utils.load_dnn_config(model_name=self.model_name)
        repo_name = config['repo_name']
        model = Wav2Vec2ForCTC.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float64)
        input_values = self.processor(input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        with torch.no_grad():
            input_values = input_values.to(self.device)
            out = self.model(input_values)
        return out
    
    def fwd_pass_tensor(self, aud_tensor):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (tensor): input tensor 'wav' input of shape (1, t) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        self.model.eval()
        logits = self.model(aud_tensor).logits
        return logits

    def transcribe(self, aud):
        """Transcribes speech audio."""
        logits = self.fwd_pass(aud)
        indexes = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(indexes)

    def batch_predictions(self, audio_batch, label_normalizer):
        """Returns prediction for the batch of audio tensors."""
        predictions = []
        with torch.no_grad():
            self.model.eval()
            
            for audio in audio_batch:
                audio = audio.to(self.device)
                indexes = torch.argmax(self.model(audio).logits, dim=-1)
                predictions.append(label_normalizer(self.processor.batch_decode(indexes)[0]))# 
        return predictions
    
###############        pretrained from Tuckute et al. 2023      ##################

class FeatureExtractorCoch(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.model_name = model_name        # cochresnet50
        config = utils.load_dnn_config(model_name=self.model_name)
        self.signal_length = config['signal_length']

        module_path = os.path.join(config['base_directory'], config['model'], 'build_network.py')
        build_network_spec = importlib.util.spec_from_file_location("build_network", module_path)
        build_network = importlib.util.module_from_spec(build_network_spec)
        build_network_spec.loader.exec_module(build_network)

        model, _, _ = build_network.main(return_metamer_layers=True)
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        num_layers = len(self.layer_ids)
    
        self.layer_rates = []
        for i in range(num_layers):
            self.layer_rates.append(self.config['layers'][i]['rate'])
        
        
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        aud_input = torch.tensor(aud, dtype=torch.float32, device=self.device)  # Convert to tensor
        self.model.eval()
        with torch.no_grad():
            out = self.model(aud_input)

        return out
    
    def extract_features_for_clip(self, audio, context_samples=0, retain_context=True):
        """
        Extracts features for a single audio clip. Audio clip must be
        sampled at 20kHz and smaller than 2s.
        For the first short clip, the context is retained, but for the later clips
        the context is not retained. 

        Args:
            aud (ndarray): single 'wav' input of shape (t,)
            context_samples (int): number of samples of context in the audio clip.
            retain_context (bool): whether to retain the context in the audio clip.

        Returns:
            features (dict): extracted features for all layers.
        """
        if audio.shape[0] > self.signal_length:
            raise ValueError(f"Audio is longer than signal length: {self.signal_length}")
            
        padding_length = self.signal_length - audio.shape[0]
        padded_audio = np.pad(audio, (0, padding_length), mode='constant')
        stim_features = self.get_features(padded_audio)
        for ii, (layer_name, feats) in enumerate(stim_features.items()):
            extra_samples_padded = int(padding_length*self.layer_rates[ii]/self.sampling_rate)
            if not retain_context:
                extra_samples_context = int(context_samples*self.layer_rates[ii]/self.sampling_rate)
            else:
                extra_samples_context = 0    
            # remove extra padded or context samples...
            num_samples = feats.shape[0]
            stim_features[layer_name] = feats[extra_samples_context:num_samples-extra_samples_padded]
        return stim_features
    

    def get_short_clips(self, audio, context_samples=0):
        """
        Splits the audio into short clips of length 2s with input context length.
        The audio is expected to be sampled at 20kHz.

        Args:
            audio (ndarray): single 'wav' input of shape (t,)
            context_samples (int): number of samples of context in the audio clip.

        Returns:
            list_clips (list): list of short audio clips each having length of 
                self.signal_length (including the context_length).
        """

        clip_length = self.signal_length - context_samples 
        list_clips = []

        num_samples = audio.shape[0]
        idx = 0
        while num_samples > clip_length:
            audio_clip = audio[idx*clip_length:(idx+1)*clip_length]
            if idx == 0:
                audio_context = np.zeros((context_samples,))
            else:
                audio_context = audio[idx*clip_length-context_samples:idx*clip_length]
            audio_clip = np.concatenate([audio_context, audio_clip])
            list_clips.append(audio_clip)
            idx += 1
            num_samples -= clip_length

        audio_clip = audio[idx*clip_length:]
        if idx == 0:
            audio_context = np.zeros((context_samples,))
        else:
            audio_context = audio[idx*clip_length-context_samples:idx*clip_length]
        audio_clip = np.concatenate([audio_context, audio_clip])
        list_clips.append(audio_clip)
        return list_clips

    
    def extract_features(self, stim_audios, sampling_rate, stim_durations=None, pad_time=None):
        """
        Returns raw features for all layers of the DNN..!

        Args:
            stim_audios (dict): dictionary of audio inputs for each sentence.
                {stim_id: audio}
            sampling_rate (int): sampling rate of the audio inputs.
            stim_durations (dict): dictionary of sentence durations.
                {stim_id: duration}
            pad_time (float): amount of padding time in seconds.

        Returns:
            dict of dict: read this as features[layer_id][stim_id]
        """
        features = {id:{} for id in self.layer_ids}
        for stim_id, audio in stim_audios.items():

            if sampling_rate != self.sampling_rate:
                n_samples = int(audio.size*self.sampling_rate/sampling_rate)
                audio = resample(audio, n_samples)
            
            if pad_time is not None:
                context_samples = int(pad_time*self.sampling_rate)
            else:
                context_samples = 0

            audio_clips = self.get_short_clips(audio, context_samples=context_samples)

            stim_features_list = []
            for ii, clip in enumerate(audio_clips):
                if ii == 0:
                    retain_context = True
                else:
                    retain_context = False
                    
                stim_features_list.append(
                    self.extract_features_for_clip(
                        clip, 
                        context_samples=context_samples, 
                        retain_context=retain_context)
                    )
            ### I need context for the first short clip, but for the later clips 
            ### I don't need it....

            for layer_id in self.layer_ids:
                layer_name = self.get_layer_name(layer_id)
                features[layer_id][stim_id] = np.concatenate([stim_feats[layer_name] for stim_feats in stim_features_list], axis=0)

            del stim_features_list
        return features

@register_feature_extractor('cochresnet50')
class CochResnet50(FeatureExtractorCoch):
    def __init__(self, shuffled=False):
        model_name = 'cochresnet50'
        super().__init__(model_name, shuffled=shuffled)

@register_feature_extractor('cochcnn9')
class CochCNN9(FeatureExtractorCoch):
    def __init__(self, shuffled=False):
        model_name = 'cochcnn9'
        super().__init__(model_name, shuffled)
        







class FeatureExtractorMERT(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        # loading our model weights
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        
        # self.model = model 
        ########################## NEED TO MAKE THIS CONSISTENT>>>##################
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float64)
        input_values = self.processor(input, sampling_rate=24000, return_tensors="pt", padding="longest").input_values  # Batch size 1
        self.model.eval()
        # with torch.no_grad():
        input_values = input_values.to(self.device)
        out = self.model(input_values)

        return out
    

class FeatureExtractorCLAP(BaseFeatureExtractor):
    def __init__(self, model_name, shuffled=False):
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        # loading our model weights
        model = ClapModel.from_pretrained("laion/larger_clap_general")
        # loading the corresponding preprocessor config
        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])

        # self.model = model 
        ########################## NEED TO MAKE THIS CONSISTENT>>>##################
    def fwd_pass(self, aud):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        input = aud.astype(np.float32)
        self.model.eval()
        input_values = self.processor(audios=input, sampling_rate=48000, return_tensors="pt")
        input_values = input_values.to(self.device)
        out = self.model.get_audio_features(**input_values)

        return out