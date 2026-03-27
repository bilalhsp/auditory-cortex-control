import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from deepspeech_pytorch.model import DeepSpeech

# local imports
from auditory_cortex import utils
from auditory_cortex import results_dir, pretrained_dir
from auditory_cortex.neural_data import UCDavisDataset



class BaseNeuralEncodingModel(ABC):
    def __init__(self, feature_extractor, model, config) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.model = model
        self.config = config

    def compute_loss(self, x, measurement):
        pred = self.forward(x)
        loss = torch.mean((pred - measurement) ** 2)
        return loss

class DeepSpeechsEncoder(torch.nn.Module):
    def __init__(self, 
                model_name='deepspeech2', trf_model=None, layer_id=0, 
                ch_idx=0, task='representation',
                device=None,
                ):
        super().__init__()
        self.model_name = model_name
        self.config = utils.load_dnn_config(model_name=self.model_name)
        self.layers_dict = {         # layer_id: layer_name
            layer['layer_id']:layer['layer_name']
            for layer in self.config['layers']
            }
        checkpoint_path = pretrained_dir / model_name / self.config['saved_checkpoint']

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only = False)
        self.model = DeepSpeech(**checkpoint['hyper_parameters'])
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.model = DeepSpeech.load_from_checkpoint(checkpoint_path=checkpoint)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
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

        self.register_hooks(layer_id)
        self.layer_features = None
        self.ch_idx = ch_idx
        self.task = task

        self.trf_layer = None
        if trf_model is not None:
            self.trf_layer, self.trf_sfreq, self.trf_features, self.trf_delays  = self._get_trf_layer(trf_model)

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
        _ = self.predict(x)

        features = self.layer_features
        if self.trf_layer is not None:

            dur = x.shape[-1]/sampling_rate
            size = UCDavisDataset.calculate_num_bins(dur, 1/self.trf_sfreq)

            feats = features.transpose(1,2)
            feats = F.interpolate(feats, size=size, mode='linear', align_corners=True)
            features = self.trf_layer(feats)[...,:-(self.trf_delays-1)]

        return features

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
        window = self.config.get('window', 'hamming')
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
