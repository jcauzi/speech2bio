import ast
import argparse
import copy
import itertools
import random
import sys
import yaml
import os
import gc
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0,'/lustre/fswork/projects/rech/jeb/uuz13rj/mount/beans')
from beans.metrics import Accuracy, MeanAveragePrecision
from beans.models import ResNetClassifier, VGGishClassifier, HuBERT_baseClassifier, AVES_bioClassifier, HuBERTClassifier, WavLMClassifier, HuBERT_baseExtractor, HuBERT_Extractor, HuBERT_Rand_Extractor, WavLM_Extractor, MeanProbe, MeanRandProbe, MeanTanhRandProbe, EchoStateNetwork, EchoStateGRUNetwork, AttentionProbe, AttentionMeanProbe, EchoStateNonlinNetwork, BiLSTMProbe, XEUS_Extractor
from beans.datasets import ClassificationDataset, RecognitionDataset
from beans.batch_dataset import LayerBatchDataset
from beans.single_dataset import LayerNonBatchDataset
from beans.chance_baseline import chance_baseline

def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets}


def spec2feats(spec):
    spec = torch.cat([
        spec.mean(dim=1),
        spec.std(dim=1),
        spec.min(dim=1)[0],
        spec.max(dim=1)[0]])
    return spec.numpy().reshape(-1)




######!!!!! CHANGING HERE !!!!!######


def check_cuda_memory(threshold_ratio=0.7):
    #return True if memory usage exceeds the threshold.
    free_memory, total_memory = torch.cuda.mem_get_info()
    used_memory = total_memory - free_memory
    return used_memory / total_memory >= threshold_ratio, used_memory / total_memory

#first function : 
def save_reps(
    args,
    dataloader_train,
    dataloader_valid,
    dataloader_test,
    save_path,
    device,
    sample_rate,
    num_labels,
    fine_tune) :

    layer_num = args.layer_num

    if args.model_type == 'HuBERT_base':
        model = HuBERT_baseExtractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'HuBERT':
        model = HuBERT_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'HuBERT_rand':
        model = HuBERT_Rand_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'WavLM':
        model = WavLM_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == "XEUS" :
        model = XEUS_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)


    datasets = {
        "train" : dataloader_train,
        "test" : dataloader_test,
        'valid' : dataloader_valid
    }
    
    #for each dataset split 
    for dataset_split in ['train', 'valid', 'test'] :
    #for dataset_split in ['train'] :

        all_labels = []
        with torch.no_grad() :
            for batch_num, (x, y) in enumerate(tqdm(datasets[dataset_split], desc=dataset_split)):

                #print("before : ", torch.cuda.memory_summary())

                model.zero_grad()

                x = x.to(device)
                y = y.to(device)
                loss, features = model(x, y)
                del loss, x
                torch.cuda.empty_cache() 
                all_labels.append(y.detach().cpu())
                del y

                #print("during : ", torch.cuda.memory_summary())


                # save batch for each layer
                for layer_index, layer_tensor in enumerate(features) :
                    layer_folder = os.path.join(save_path, dataset_split, f"layer_{layer_index}")
                    os.makedirs(layer_folder, exist_ok=True)
                    batch_file_path = os.path.join(layer_folder, f"batch_{batch_num}.pt")
                    torch.save(layer_tensor, batch_file_path)

                    del layer_tensor
                    gc.collect()
                    torch.cuda.empty_cache() 

                    #print("inside : ", torch.cuda.memory_summary())

                    
                torch.cuda.synchronize()
                del features  
                gc.collect()  
                torch.cuda.empty_cache()

                #print("after : ", torch.cuda.memory_summary())

        
        #add labels
        all_labels = torch.cat(all_labels, dim=0)
        labels_path = os.path.join(save_path, dataset_split, "labels.pt")
        torch.save(all_labels, labels_path)
        del all_labels
        gc.collect()
        torch.cuda.empty_cache()

def save_reps_individual(
    args,
    dataloader_train,
    dataloader_valid,
    dataloader_test,
    save_path,
    device,
    sample_rate,
    num_labels,
    fine_tune):

    layer_num = args.layer_num

    if args.model_type == 'HuBERT_base':
        model = HuBERT_baseExtractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'HuBERT':
        model = HuBERT_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'HuBERT_rand':
        model = HuBERT_Rand_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == 'WavLM':
        model = WavLM_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)
    elif args.model_type == "XEUS" :
        model = XEUS_Extractor(
            sample_rate=sample_rate,
            num_classes=num_labels,
            multi_label=(args.task=='detection'),
            fine_tune=fine_tune,
            layer_num=layer_num).to(device)


    datasets = {
        "train": dataloader_train,
        "valid": dataloader_valid,
        "test": dataloader_test
    }

    for dataset_split in ["train", "valid", "test"]:
    #for dataset_split in ["train", "valid"]:

        label_list = []
        file_index = 0  # Track individual file numbering

        with torch.no_grad():
            for batch_num, (x, y) in enumerate(tqdm(datasets[dataset_split], desc=dataset_split)):
                model.zero_grad()

                x = x.to(device)
                y = y.to(device)
                loss, features = model(x, y)
                del loss, x
                torch.cuda.empty_cache()

                y = y.detach().cpu()
                label_list.append(y)

                # Save individual examples
                for i in range(y.shape[0]):  # Iterate through batch
                    for layer_index, layer_tensor in enumerate(features):
                        layer_folder = os.path.join(save_path, dataset_split, f"layer_{layer_index}")
                        os.makedirs(layer_folder, exist_ok=True)

                        # Save each example separately
                        file_path = os.path.join(layer_folder, f"sample_{file_index}.pt")
                        if not os.path.exists(file_path):
                            torch.save(layer_tensor[i].cpu(), file_path)

                    file_index += 1  # Increment index for each example

                del y, features
                gc.collect()
                torch.cuda.empty_cache()

        # Save all labels in a single file
        all_labels = torch.cat(label_list, dim=0)
        labels_path = os.path.join(save_path, dataset_split, "labels.pt")
        torch.save(all_labels, labels_path)
        del all_labels
        gc.collect()
        torch.cuda.empty_cache()




# def load_reps(args, save_path, dataset_split, shuffle=False):

#     layer_num = args.layer_num
#     batch_size=args.batch_size

#     # Define the paths
#     layer_folder = os.path.join(save_path, dataset_split, f"layer_{layer_num}")
#     labels_path = os.path.join(save_path, dataset_split, "labels.pt")

#     # Load features and labels
#     if not os.path.exists(layer_folder):
#         raise FileNotFoundError(f"Layer folder not found: {layer_folder}")
#     if not os.path.exists(labels_path):
#         raise FileNotFoundError(f"Labels file not found: {labels_path}")

#     layer_files = sorted(
#         [os.path.join(layer_folder, f) for f in os.listdir(layer_folder) if f.startswith("batch_")],
#         key=lambda x: int(x.split("_")[-1].split(".")[0])  # Sort by batch number
#     )
#     if not layer_files:
#         raise FileNotFoundError(f"No batch files found in layer folder: {layer_folder}")

#     features = [torch.load(file) for file in layer_files]
#     features = torch.cat(features, dim=0)  # Concatenate all batch tensors

#     # Load labels
#     labels = torch.load(labels_path)

#     print(f"Features shape: {features.shape}")
#     print(f"Labels shape: {labels.shape}")

#     # Create a TensorDataset and DataLoader
#     dataset = TensorDataset(features, labels)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#     return dataloader



def eval_sklearn_model(model_and_scaler, dataloader, num_labels, metric_factory):
    total_loss = 0.
    metric = metric_factory()
    model, scaler = model_and_scaler

    for x, y in dataloader:
        xs = [spec2feats(x[i]) for i in range(x.shape[0])]
        xs_scaled = scaler.transform(xs)
        pred = model.predict(xs_scaled)
        if isinstance(model, MultiOutputClassifier):
            pred = torch.tensor(pred)
        else:
            pred = F.one_hot(torch.tensor(pred), num_classes=num_labels)
        metric.update(pred, y)

    return total_loss, metric.get_primary_metric()


def train_sklearn_model(args, dataloader_train, dataloader_valid, num_labels, metric_factory, log_file):
    print(f'Building training data ...', file=sys.stderr)

    xs = []
    ys = []
    for x, y in dataloader_train:
        xs.extend(spec2feats(x[i]) for i in range(x.shape[0]))
        ys.extend(y[i].numpy() for i in range(y.shape[0]))

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs)
    print(f"Num. features = {xs_scaled[0].shape}, num. instances = {len(xs_scaled)}", file=sys.stderr)

    params = ast.literal_eval(args.params)
    assert(isinstance(params, dict))
    param_list = [[(k, v) for v in vs] for k, vs in params.items()]
    param_combinations = itertools.product(*param_list)

    valid_metric_best = 0.
    best_model = None

    for extra_params in param_combinations:
        extra_params = dict(extra_params)
        print(f'Fitting data (params: {extra_params})...', file=sys.stderr)

        if args.model_type == 'lr':
            model = LogisticRegression(max_iter=1_000, **extra_params)
        elif args.model_type == 'svm':
            model = SVC(**extra_params)
        elif args.model_type == 'decisiontree':
            model = DecisionTreeClassifier(**extra_params)
        elif args.model_type == 'gbdt':
            model = GradientBoostingClassifier(**extra_params)
        elif args.model_type == 'xgboost':
            model = XGBClassifier(n_jobs=4, **extra_params)

        if args.task == 'detection':
            model = MultiOutputClassifier(model)

        model.fit(xs_scaled, ys)

        _, valid_metric = eval_sklearn_model(
            model_and_scaler=(model, scaler),
            dataloader=dataloader_valid,
            num_labels=num_labels,
            metric_factory=metric_factory)

        if valid_metric > valid_metric_best:
            best_model = model
            valid_metric_best = valid_metric

        print({
            'extra_params': extra_params,
            'valid': {
                'metric': valid_metric
            }}, file=log_file)

    return (best_model, scaler), valid_metric_best


def eval_pytorch_model(model, dataloader, metric_factory, device, desc):
    model.eval()
    total_loss = 0.
    steps = 0
    metric = metric_factory()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            #!!!!!! HACKY SQUEEZE !!!!!!!!!!!!!!!!!!!!!!
            x = x.squeeze(0)
            y = y.squeeze(0)

            loss, logits = model(x, y)
            total_loss += loss.cpu().item()
            steps += 1

            metric.update(logits, y)
            torch.cuda.empty_cache()

    total_loss /= steps

    return total_loss, metric.get_primary_metric()

# def save_checkpoint(model, optimizer, epoch,  loss, path="checkpoint.pth"):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }
#     torch.save(checkpoint, path)

# def load_checkpoint(model, optimizer, path="checkpoint.pth"):
#     if torch.cuda.is_available():
#         checkpoint = torch.load(path)  # Use CUDA if available
#     else:
#         checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Load on CPU

#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     return epoch, loss

def train_pytorch_model(
    args,
    dataloader_train,
    dataloader_valid,
    dataloader_test,
    num_labels,
    metric_factory,
    sample_rate,
    device,
    log_file,
    in_features,
    n_frames,
    fine_tune=True,
    layer_num=-1):

    lrs = ast.literal_eval(args.lrs)
    assert isinstance(lrs, list)

    valid_metric_best = 0.
    best_model = None

    for lr in lrs:
        print(f"lr = {lr}" , file=log_file)

        if args.probe == "linear" :
            model = MeanProbe(
                sample_rate=sample_rate,
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection'),
                fine_tune=fine_tune,
                layer_num=layer_num).to(device)
            print("model mean linear")
        if args.probe == "linear_max" :
            model = MeanProbe(
                sample_rate=sample_rate,
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection'),
                fine_tune=fine_tune,
                layer_num=layer_num).to(device)
            print("model max linear")
        elif args.probe == "rand_linear" :
            model = MeanRandProbe(
                sample_rate=sample_rate,
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection'),
                fine_tune=fine_tune,
                layer_num=layer_num).to(device)
            print("model rand linear")
        elif args.probe == "rand_tanh_linear" :
            model = MeanTanhRandProbe(
                sample_rate=sample_rate,
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection'),
                fine_tune=fine_tune,
                layer_num=layer_num).to(device)
            print("model rand tanh linear")
        elif args.probe == "echo_state" :
            model = EchoStateNetwork(
                embedding_size=in_features,
                n_frames=n_frames,
                num_classes=num_labels,
                reservoir_size=4048,
                multi_label=(args.task=='detection')).to(device)
            print("model echo state network")
        elif args.probe == "echo_state_gru" :
            model = EchoStateGRUNetwork(
                embedding_size=in_features,
                n_frames=n_frames,
                hidden_size=256,
                num_classes=num_labels,
                reservoir_size=2048,
                multi_label=(args.task=='detection')).to(device)
            print("model echo state gru")
        elif args.probe == "attention_probe" :
            model = AttentionProbe(
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection')).to(device)
            print("model attention linear")
        elif args.probe == "attention_mean_probe" :
            model = AttentionMeanProbe(
                in_features=in_features,
                num_classes=num_labels,
                multi_label=(args.task=='detection')).to(device)
            print("model attention mean")
        elif args.probe == "echo_state_nonlin" :
            model = EchoStateNonlinNetwork(
                embedding_size=in_features,
                n_frames=n_frames,
                hidden_size=612,
                num_classes=num_labels,
                reservoir_size=2048,
                multi_label=(args.task=='detection')).to(device)
            print("model attention mean")
        elif args.probe == "bilstm" :
            model = BiLSTMProbe(
                in_features=in_features,
                hidden_size=256,
                num_classes=num_labels,
                multi_label=(args.task=='detection')).to(device)

        #adding weight decay 
        #optimizer = optim.Adam(params=model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        epochs_without_improvement=0


        total_params = sum(p.numel() for p in model.parameters())

        # Calculate only the trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("TOTAL PARAMS : ", total_params)
        print("TRAINABLE PARAMS : ", trainable_params)

        print("fine tune arg in train : ", fine_tune)

        for epoch in range(args.epochs):
            print(f'epoch = {epoch}', file=sys.stderr)

            model.train()

            train_loss = 0.
            train_steps = 0
            train_metric = metric_factory()

            for x, y in tqdm(dataloader_train, desc='train'):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)


    #!!!!!!! HACKY SQUEEZING !!!!!!!!!!!

                x = x.squeeze(0)
                y = y.squeeze(0)

                #print(f"shape of x : {x.shape}, shape of y : {y.shape}")

                loss, logits = model(x, y)

                #print(f"logits shape : {logits.shape}")

                loss.backward()

                optimizer.step()

                train_loss += loss.detach()
                train_steps += 1

                train_metric.update(logits, y)

                #might slow down training : 
                torch.cuda.empty_cache()

            valid_loss, valid_metric = eval_pytorch_model(
                model=model,
                dataloader=dataloader_valid,
                metric_factory=metric_factory,
                device=device,
                desc='valid')
            
            patience = 50

            if valid_metric > valid_metric_best:
                valid_metric_best = valid_metric
                best_model = copy.deepcopy(model)
                epochs_without_improvement = 0
                print("Validation metric improved, saving model.", file=sys.stderr)
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s).", file=sys.stderr)
            
                # early stopping condition
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

            print({
                'epoch': epoch,
                'train': {
                    'loss': (train_loss / train_steps).cpu().item(),
                    'metric': train_metric.get_metric(),
                },
                'valid': {
                    'loss': valid_loss,
                    'metric': valid_metric
                }
            }, file=log_file)
            log_file.flush()
    
            #might be better (for each epoch)
            torch.cuda.empty_cache()


        _, test_metric = eval_pytorch_model(
                model=model,
                dataloader=dataloader_test,
                metric_factory=metric_factory,
                device=device,
                desc='test')

        print(
        'valid_metric_best = ', valid_metric_best,
        'test_metric = ', test_metric,
        file=log_file)


    return best_model, valid_metric_best


def main():
    datasets = read_datasets('datasets.yml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lrs', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--task', choices=['classification', 'detection'])
    parser.add_argument('--model-type', choices=[
        'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost',
        'resnet18', 'resnet18-pretrained',
        'resnet50', 'resnet50-pretrained',
        'resnet152', 'resnet152-pretrained',
        'vggish', 'HuBERT_base', 'AVES_bio', 'HuBERT', 'WavLM', 'XEUS', 'HuBERT_rand'])
    parser.add_argument('--dataset', choices=datasets.keys())
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--stop-shuffle', action='store_true')
    parser.add_argument('--log-path', type=str)
    parser.add_argument('--fine_tune', type=str)
    parser.add_argument('--layer_num', type=int)
    parser.add_argument('--probe', type=str)
    args = parser.parse_args()

    #Transforming str argument into bool
    fine_tune = args.fine_tune == "True"

    print("FINE TUNE ARG IN EVALUATE after solution : ", fine_tune, type(fine_tune))

    torch.random.manual_seed(42)
    random.seed(42)
    if args.log_path:
        log_file = open(args.log_path, mode='w')
    else:
        log_file = sys.stderr

    device = torch.device('cuda:0')

    if args.model_type == 'vggish' :
        feature_type = 'vggish'
    elif args.model_type.startswith('HuBERT') :
        feature_type = 'waveform'
    elif args.model_type == 'WavLM' :
        feature_type = 'waveform'
    elif args.model_type == 'XEUS' :
        feature_type = 'waveform'
    elif args.model_type == 'AVES_bio' :
        feature_type = 'waveform'
    elif args.model_type.startswith('resnet'):
        feature_type = 'melspectrogram'
    else:
        feature_type = 'mfcc'

    dataset = datasets[args.dataset]
    num_labels = dataset['num_labels']

    if dataset['type'] == 'classification':
        dataset_train = ClassificationDataset(
            metadata_path=dataset['train_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)
        dataset_valid = ClassificationDataset(
            metadata_path=dataset['valid_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)
        dataset_test = ClassificationDataset(
            metadata_path=dataset['test_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)

    elif dataset['type'] == 'detection':
        dataset_train = RecognitionDataset(
            metadata_path=dataset['train_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
        dataset_valid = RecognitionDataset(
            metadata_path=dataset['valid_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
        dataset_test = RecognitionDataset(
            metadata_path=dataset['test_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
    else:
        raise ValueError(f"Invalid dataset type: {dataset['type']}")
    
    print("num_workers : ", args.num_workers)

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=not args.stop_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True)
    if dataset_test is not None:
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True)
    else:
        dataloader_test = None



    # def compute_class_proportions(dataloader, num_classes):
    #     class_counts = torch.zeros(num_classes)
    #     total_labels = 0

    #     for _ , labels in dataloader:
    #         for label in labels:
    #             class_counts[label] += 1
    #         total_labels += labels.size(0)

    #     class_proportions = class_counts / total_labels
    #     return class_proportions

    # num_classes = 50  # Adjust this based on your dataset
    # proportions = compute_class_proportions(dataloader_test, num_classes)

    # print("Class Proportions:", proportions)


###!!!!!!! VERSION 1 !!!!!!!!######


    # #Extract or load representations :
    # representation_path = f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}/{args.model_type}/"
    # #if not os.path.isdir(os.path.join(representation_path, "train", f"layer_{args.layer_num}")):
    # if not os.path.isfile(os.path.join(representation_path, "train", "labels.pt")):
    #     print(f"The folder doesn't exist / extracting representations for {args.model_type}")
    #     save_reps(args,
    #               dataloader_train,
    #               dataloader_valid,
    #               dataloader_test,
    #               representation_path,
    #               device,
    #               dataset.get('sample_rate', 16000),
    #               num_labels,
    #               fine_tune,
    #                )
    # else : print(f"Reps already extracted for {args.model_type}")


    # layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type, "train")
    # labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type, "train", "labels.pt")
    # data = LayerBatchDataset(layer_folder, labels_path, args.layer_num)
    # dataloader_train = DataLoader(data, batch_size=1, shuffle=False)  # Load one batch at a time

    # layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type,  "valid")
    # labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type, "valid", f"labels.pt")
    # data = LayerBatchDataset(layer_folder, labels_path, args.layer_num)
    # dataloader_valid = DataLoader(data, batch_size=1, shuffle=False)  # Load one batch at a time

    # layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type, "test")
    # labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}", args.model_type, "test", f"labels.pt")
    # data = LayerBatchDataset(layer_folder, labels_path, args.layer_num)
    # dataloader_test = DataLoader(data, batch_size=1, shuffle=False)  # Load one batch at a time

####!!!!!!! VERSION 2 !!!!!!!!######


 #Extract or load representations :
    representation_path = f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test/{args.model_type}/"
    if not os.path.isdir(os.path.join(representation_path, "train", f"layer_{args.layer_num}")):
    #if not os.path.isfile(os.path.join(representation_path, "train", "labels.pt")):
    #print(os.path.join(representation_path, "train", f"layer_{args.layer_num}"))
    #print(os.listdir(os.path.join(representation_path, "train", f"layer_{args.layer_num}")))
    #if not os.listdir(os.path.join(representation_path, "train", f"layer_{args.layer_num}")):
    #if True :
        print(f"The folder doesn't exist / extracting single representations for {args.model_type}")
        save_reps_individual(args,
                  dataloader_train,
                  dataloader_valid,
                  dataloader_test,
                  representation_path,
                  device,
                  dataset.get('sample_rate', 16000),
                  num_labels,
                  fine_tune,
                   )
    else : print(f"Reps already extracted for {args.model_type}")

    layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type, "train")
    labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type, "train", "labels.pt")
    data = LayerNonBatchDataset(layer_folder, labels_path, args.layer_num)
    dataloader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False)  # Load one batch at a time

    layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type,  "valid")
    labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type, "valid", f"labels.pt")
    data = LayerNonBatchDataset(layer_folder, labels_path, args.layer_num)
    dataloader_valid = DataLoader(data, batch_size=args.batch_size, shuffle=False)  # Load one batch at a time

    layer_folder = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type, "test")
    labels_path = os.path.join(f"/lustre/fsn1/projects/rech/jeb/uuz13rj/beans/{args.dataset}_single_test", args.model_type, "test", f"labels.pt")
    data = LayerNonBatchDataset(layer_folder, labels_path, args.layer_num)
    dataloader_test = DataLoader(data, batch_size=args.batch_size, shuffle=False)  # Load one batch at a time













    #extract embedding_size for probe initilization
    for features, labels in dataloader_train:
        features = features.squeeze(0)
        print(f"features shape : {features.shape}")
        in_features = features.shape[-1]  # Extract the last dimension of features
        print(f"Embedding size: {in_features}")
        n_frames = features.shape[-2]
        print(f"Number of frames : {n_frames}")
        break



    if args.task == 'classification':
        Metric = Accuracy
    elif args.task == 'detection':
        Metric = MeanAveragePrecision

    if args.model_type in {'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost'}:
        model_and_scaler, valid_metric_best = train_sklearn_model(
            args=args,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            num_labels=num_labels,
            metric_factory=Metric,
            log_file=log_file)

        if dataloader_test is not None:
            _, test_metric = eval_sklearn_model(
                model_and_scaler=model_and_scaler,
                dataloader=dataloader_test,
                num_labels=num_labels,
                metric_factory=Metric)

    else:
        model, valid_metric_best = train_pytorch_model(
            args=args,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            dataloader_test=dataloader_test,
            num_labels=num_labels,
            metric_factory=Metric,
            sample_rate=dataset.get('sample_rate', 16000),
            device=device,
            log_file=log_file,
            in_features=in_features,
            n_frames=n_frames,
            fine_tune=fine_tune,
            layer_num=args.layer_num
            )

        if dataloader_test is not None:
            _, test_metric = eval_pytorch_model(
                model=model,
                dataloader=dataloader_test,
                metric_factory=Metric,
                device=device,
                desc='test')

    print(
        'valid_metric_best = ', valid_metric_best,
        'test_metric = ', test_metric,
        file=log_file)

    if args.log_path:
        log_file.close()

if __name__ == '__main__':
    main()
