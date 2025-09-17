import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import gc
import copy
from saved_models.torchvggish.torchvggish.vggish import VGGish
from transformers import HubertModel, Wav2Vec2FeatureExtractor, WavLMModel, HubertConfig
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq
print("imports done")
from espnet2.tasks.ssl import SSLTask
print("last import")
#import soundfile as sf

class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
                loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        #initial online implementation:
        #self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    
        #local implementation of vggish pretrained (VGGish architecture from the git and state_dict in saved_models/)
        self.vggish = VGGish(["fake argument"], pretrained=False)
        self.vggish.load_state_dict(torch.load('saved_models/vggish.pth'))
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class MeanProbe(nn.Module):
    def __init__(self, sample_rate, in_features, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()


    def forward(self, x, y=None):
        mean_embeddings = torch.mean(x, dim=1)
        #mean_embeddings, _ = torch.max(x, dim=1)
        logits = self.linear(mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits
    
class BiLSTMProbe(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes=None, multi_label=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Define the BiLSTM
        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=1,  # You can increase this for deeper networks
            batch_first=True,
            bidirectional=True
        )
        
        # Define the linear layer
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

        # Loss function
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x shape: (batch_size, n_frames, in_features)
        
        # Pass through the BiLSTM
        bilstm_out, (h_n, c_n) = self.bilstm(x)  # bilstm_out shape: (batch_size, n_frames, hidden_size * 2)
        
        # Aggregate the BiLSTM output
        # Here we use the last hidden state from both directions (concatenated)
        # Alternatively, you can use bilstm_out[:, -1, :] for only the last output
        final_representation = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # shape: (batch_size, hidden_size * 2)
        
        # Pass through the linear layer
        logits = self.linear(final_representation)
        
        # Compute loss if labels are provided
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        
        return loss, logits

class AttentionProbe(nn.Module): 
    def __init__(self, in_features, num_classes=None, multi_label=False, k=200) :
        super().__init__()

        self.k = k

        self.in_features = in_features

        #temporal attention wieghts :
        self.attention_weights = nn.Linear(in_features, 1)

        #probe :
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        
        attention_scores = self.attention_weights(x)#.squeeze(-1) 

        # # Get the top-k attention scores and their indices
        # topk_scores, topk_indices = torch.topk(attention_scores, self.k, dim=1) 

        # # Create a mask to zero-out non-top-k scores
        # attention_mask = torch.zeros_like(attention_scores)
        # attention_mask.scatter_(1, topk_indices, topk_scores)

        # # Normalize the top-k scores for soft attention
        # attention_scores = attention_mask / (attention_mask.sum(dim=1, keepdim=True) + 1e-8)

        # attention_scores = torch.softmax(attention_scores, dim=1) # (batch_size, n_frames, 1)

        #weighted sum of embeddings : 
        #weighted_embeddings = torch.sum(x * attention_scores.unsqueeze(-1), dim=1) #(batch_size, in_features) same shape as mean pooling
        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_embeddings = torch.sum(x * attention_weights, dim=1) #(batch_size, in_features) same shape as mean pooling


        logits = self.linear(weighted_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits

class AttentionMeanProbe(nn.Module): 
    def __init__(self, in_features, num_classes=None, multi_label=False) :
        super().__init__()

        self.in_features = in_features

        #temporal attention wieghts :
        self.attention_weights = nn.Linear(in_features, 1)

        #probe :
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):

        mean_embeddings = torch.mean(x, dim=1)

        attention_scores = self.attention_weights(x) 
        attention_scores = torch.softmax(attention_scores, dim=1) # (batch_size, n_frames, 1)

        #weighted sum of embeddings : 
        weighted_embeddings = torch.sum(x * attention_scores, dim=1) #(batch_size, in_features) same shape as mean pooling

        logits = self.linear(weighted_embeddings + mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits


class MeanRandProbe(nn.Module):
    def __init__(self, sample_rate, in_features, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)
        self.random = nn.Linear(in_features=in_features, out_features=in_features)

        for param in self.random.parameters():
            param.requires_grad = False

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate
        self.layer_num = layer_num

    def forward(self, x, y=None):
        mean_embeddings = torch.mean(x, dim=1)
        mean_embeddings = self.random(mean_embeddings)
        logits = self.linear(mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits
    
class MeanTanhRandProbe(nn.Module):
    def __init__(self, sample_rate, in_features, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)
        self.random = nn.Linear(in_features=in_features, out_features=in_features)
        self.tanh = nn.Tanh() 

        for param in self.random.parameters():
            param.requires_grad = False

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        mean_embeddings = torch.mean(x, dim=1)
        mean_embeddings = self.random(mean_embeddings)
        mean_embeddings = self.tanh(mean_embeddings)
        logits = self.linear(mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits


class EchoStateNetwork(nn.Module):
    def __init__(self, embedding_size, n_frames, num_classes, reservoir_size, sparsity=0.05, spectral_radius=0.8, multi_label=False):
        """
            n_frames: time steps in input sequences.
            reservoir_size: neurons in the reservoir.
            sparsity: fraction of non-zero connections in the reservoir
            spectral_radius: scaling factor for the reservoir's weight matrix
        """
        super(EchoStateNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.n_frames = n_frames
        self.num_classes = num_classes
        self.reservoir_size = reservoir_size
        
        #initialize the reservoir with hyper parameters :
        self.reservoir = nn.Parameter(
            self._initialize_reservoir(reservoir_size, sparsity, spectral_radius),
            requires_grad=False
        )
        
        #initialize input weights for the reservoir :
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, embedding_size) * 0.1,
            requires_grad=False
        )
        
        #probe 
        #self.linear = nn.Linear(reservoir_size, num_classes)

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout with 50% probability
            nn.Linear(reservoir_size, num_classes)
        )

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def _initialize_reservoir(self, size, sparsity, spectral_radius):
        # reservoir is a square matrix 
        reservoir = torch.randn(size, size) * (torch.rand(size, size) < sparsity).float()
        #normalizing spectral radius :
        eigenvalues = torch.linalg.eigvals(reservoir)
        max_eigenvalue = torch.max(eigenvalues.abs())
        reservoir *= spectral_radius / max_eigenvalue
        return reservoir
    
    def forward(self, x, y=None):
        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        #iterate over frames 
        for f in range(self.n_frames):
            input_f = x[:, f, :]  #shape: (batch_size, embedding_size)
            state = torch.tanh(
                torch.matmul(state, self.reservoir.T) + torch.matmul(input_f, self.input_weights.T)
            )
        
        logits = self.linear(state)  # shape: (batch_size, num_classes)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits



    #ADDING A GRU : 
    # - Helps processing long sequences: For input sequences with a large number of frames (e.g., 688), where long-term dependencies matter.
    # - Helps with complex dynamics: When the frozen ESN reservoir struggles to fully capture the temporal patterns of the input.
    # - Helps with overfitting problems: The GRU's gating mechanism can help selectively filter noisy or irrelevant features.



class EchoStateGRUNetwork(nn.Module):
    def __init__(self, embedding_size, n_frames, hidden_size, num_classes, reservoir_size, sparsity=0.1, spectral_radius=9.9, multi_label=False):

        super(EchoStateGRUNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.n_frames = n_frames
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reservoir_size = reservoir_size
        
        #initialize the reservoir with hyper parameters :
        self.reservoir = nn.Parameter(
            self._initialize_reservoir(reservoir_size, sparsity, spectral_radius),
            requires_grad=False
        )
        
        #initialize input weights for the reservoir :
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, embedding_size) * 0.1,
            requires_grad=False
        )
        
        #gru
        self.gru = nn.GRU(reservoir_size, hidden_size, batch_first=True)
        
        
        #probe 
        #self.linear = nn.Linear(reservoir_size, num_classes)

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout with 50% probability
            nn.Linear(hidden_size, num_classes)
        )

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def _initialize_reservoir(self, size, sparsity, spectral_radius):
        # reservoir is a square matrix 
        reservoir = torch.randn(size, size) * (torch.rand(size, size) < sparsity).float()
        #normalizing spectral radius :
        eigenvalues = torch.linalg.eigvals(reservoir)
        max_eigenvalue = torch.max(eigenvalues.abs())
        reservoir *= spectral_radius / max_eigenvalue
        return reservoir
    
    def forward(self, x, y=None):

        batch_size, seq_length, _ = x.shape
        state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        reservoir_states = []

        # generate reservoir states for each time step
        for t in range(seq_length):
            input_t = x[:, t, :]  #  (batch_size, input_size)
            state = torch.tanh(
                torch.mm(input_t, self.input_weights.T) +
                torch.mm(state, self.reservoir.T)
            )
            reservoir_states.append(state)

        # xoncatenate reservoir states 
        reservoir_states = torch.stack(reservoir_states, dim=1)  # (batch_size, seq_length, reservoir_size)

        gru_out, _ = self.gru(reservoir_states)  #(batch_size, seq_length, hidden_size)

        # last hidden state for classification
        final_state = gru_out[:, -1, :]  #  (batch_size, hidden_size)

        logits = self.linear(final_state)  # shape: (batch_size, num_classes)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits

class EchoStateNonlinNetwork(nn.Module):
    def __init__(self, embedding_size, n_frames, hidden_size, num_classes, reservoir_size, sparsity=0.1, spectral_radius=0.8, multi_label=False):

        super(EchoStateNonlinNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.n_frames = n_frames
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reservoir_size = reservoir_size
        
        #initialize the reservoir with hyper parameters :
        self.reservoir = nn.Parameter(
            self._initialize_reservoir(reservoir_size, sparsity, spectral_radius),
            requires_grad=False
        )
        
        #initialize input weights for the reservoir :
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, embedding_size) * 0.1,
            requires_grad=False
        )        
        
        #probe 
        #self.linear = nn.Linear(reservoir_size, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(reservoir_size, hidden_size),  # Hidden layer
            nn.ReLU(),                              # Non-linear activation
            nn.Linear(hidden_size, num_classes)     # Output layer
        )

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def _initialize_reservoir(self, size, sparsity, spectral_radius):
        # reservoir is a square matrix 
        reservoir = torch.randn(size, size) * (torch.rand(size, size) < sparsity).float()
        #normalizing spectral radius :
        eigenvalues = torch.linalg.eigvals(reservoir)
        max_eigenvalue = torch.max(eigenvalues.abs())
        reservoir *= spectral_radius / max_eigenvalue
        return reservoir
    
    def forward(self, x, y=None):

        batch_size, seq_length, _ = x.shape
        state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        reservoir_states = []

        # Generate reservoir states for each time step
        for t in range(seq_length):
            input_t = x[:, t, :]  # Shape: (batch_size, input_size)
            state = torch.tanh(
                torch.mm(input_t, self.input_weights.T) +
                torch.mm(state, self.reservoir.T)
            )
            reservoir_states.append(state)

        # Stack reservoir states into a single tensor: (batch_size, seq_length, reservoir_size)
        reservoir_states = torch.stack(reservoir_states, dim=1)

        # Summarize reservoir states into a fixed-size vector
        # Option 1: Use the final state (state at the last time step)
        final_state = reservoir_states[:, -1, :]  # Shape: (batch_size, reservoir_size)

        # Option 2: Use mean pooling over the sequence
        # final_state = torch.mean(reservoir_states, dim=1)  # Shape: (batch_size, reservoir_size)

        # Option 3: Use max pooling over the sequence
        # final_state, _ = torch.max(reservoir_states, dim=1)  # Shape: (batch_size, reservoir_size)

        # Pass summarized state to the classifier
        logits = self.classifier(final_state)  # Shape: (batch_size, num_classes)

        # Compute loss if labels are provided
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class HuBERT_baseClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/hubert_base"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.hubert = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)

        self.linear = nn.Linear(in_features=768, out_features=num_classes)

        #Freeze HuBERT model
        if not fine_tune :
            for param in self.hubert.parameters():
                param.requires_grad = False
        print(multi_label)
        print("layer_num = ", layer_num)
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate
        self.layer_num = layer_num

    def forward(self, x, y=None):
        out = self.hubert(x, output_hidden_states=True)
        mean_embeddings = torch.mean(out.hidden_states[self.layer_num], dim=1)
        logits = self.linear(mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits

class HuBERT_baseExtractor(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/hubert_base"
        self.hubert = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)

        #Freeze HuBERT model
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("MODEL FROZEN")

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        out = self.hubert(x, output_hidden_states=True)
        all_layers = out.hidden_states[13:]
        loss = None
        return loss, all_layers

class HuBERT_Extractor(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/hubert_large-ll60k"
        self.hubert = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)

        #Freeze HuBERT model
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("MODEL FROZEN")

    def forward(self, x, y=None):
        out = self.hubert(x, output_hidden_states=True)
        all_layers = out.hidden_states
        loss = None
        return loss, all_layers

class HuBERT_Rand_Extractor(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/hubert_large-ll60k"
        #self.hubert = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)
        config = HubertConfig.from_json_file("saved_models/hubert_large-ll60k/config.json")
        self.hubert = HubertModel(config)

        #Freeze HuBERT model
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("MODEL FROZEN - random init")

    def forward(self, x, y=None):
        out = self.hubert(x, output_hidden_states=True)
        all_layers = out.hidden_states
        loss = None
        return loss, all_layers

class WavLM_Extractor(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/wavlm-large/"
        self.wavlm = WavLMModel.from_pretrained(MODEL_DIR)

        #Freeze HuBERT model
        for param in self.wavlm.parameters():
            param.requires_grad = False
        print("MODEL FROZEN")

    def forward(self, x, y=None):
        out = self.wavlm(x, output_hidden_states=True)
        #use [:1] to get only the layer 0 with wrapping (cannot extract layer if not in list because it would reduce one dimension...)
        all_layers = copy.deepcopy(out.hidden_states)
        del out
        gc.collect()
        torch.cuda.empty_cache() 
        loss = None
        return loss, all_layers
    
class XEUS_Extractor(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        ckpt_dir = "saved_models/XEUS/model/xeus_checkpoint.pth"

        self.xeus_model, self.xeus_train_args = SSLTask.build_model_from_file(None, ckpt_dir)

        #Freeze HuBERT model
        for param in self.xeus_model.parameters():
            param.requires_grad = False
        print("MODEL FROZEN")

    def forward(self, x, y=None):
        #print("shape of x[0]", x[0].shape)
        #print("shape of x", x.shape)
        speech_lengths = torch.tensor([waveform.shape[0] for waveform in x], dtype=torch.int32)
        #speech_lengths = speech_lengths.to(dtype=torch.int32)
        #print(speech_lengths)
        #print(f"Speech lengths shape: {speech_lengths.shape}, dtype: {speech_lengths.dtype}")
        #out = self.xeus_model(x, speech_lengths, output_hidden_states=True)
        #print(x)
        out = self.xeus_model.encode(x, speech_lengths, use_mask=False, use_final_output=False)[0]
        #print(len(out))
        #print(out[-1].shape)
        #print(out.shape)
        #use [:1] to get only the layer 0 with wrapping (cannot extract layer if not in list because it would reduce one dimension...)
        loss = None
        return loss, out

class AVES_bioClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        model_file = 'saved_models/aves-base-bio.pt'
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        original = models[0]
        self.aves = import_fairseq_model(original)

        self.linear = nn.Linear(in_features=768, out_features=num_classes)

        # Freeze AVES model
        if not fine_tune :
            for param in self.aves.parameters():
                param.requires_grad = False
                print("not fine tuning")
        else : print("fine tuning")

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate
        self.layer_num = layer_num

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        
        features, _ = self.aves.extract_features(x)
        hidden_state = features[self.layer_num]

        mean_embedding = torch.mean(hidden_state, dim=1)
        
        logits = self.linear(mean_embedding)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class HuBERTClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/hubert_large-ll60k"
        self.hubert = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

        #Freeze HuBERT model
        if not fine_tune :
            for param in self.hubert.parameters():
                param.requires_grad = False

        #print(multi_label)
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate
        self.layer_num = layer_num

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        out = self.hubert(x, output_hidden_states=True)
        mean_embeddings = torch.mean(out.hidden_states[self.layer_num], dim=1)
        logits = self.linear(mean_embeddings)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        return loss, logits
    
class WavLMClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False, fine_tune=True, layer_num=-1):
        super().__init__()

        MODEL_DIR = "saved_models/wavlm-large/"
        self.wavlm = WavLMModel.from_pretrained(MODEL_DIR)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

        #Freeze wavlm model
        if not fine_tune :
            for param in self.wavlm.parameters():
                param.requires_grad = False

        #print(multi_label)
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate
        self.layer_num = layer_num

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        out = self.wavlm(x, output_hidden_states=True)
        mean_embeddings = torch.mean(out.hidden_states[self.layer_num], dim=1)
        logits = self.linear(mean_embeddings)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits