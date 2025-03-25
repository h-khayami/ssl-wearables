import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
from torchvision import transforms
import pathlib
import json

# SSL net
from sslearning.models.accNet import cnn1, SSLNET, Resnet, EncoderMLP
from sslearning.scores import classification_scores, classification_report
import copy
from sklearn import preprocessing
from sslearning.data.data_loader import NormalDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torchsummary import summary # print model summary
from sslearning.pytorchtools import EarlyStopping
from sslearning.data.datautils import RandomSwitchAxis, RotationAxis
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
import collections
from hydra.utils import get_original_cwd

from dotenv import load_dotenv
from discord_webhook import DiscordWebhook
from data_parsing.Dataset_Label_Mapper import DatasetLabelMapper

"""
python downstream_task_evaluation.py -m data=rowlands_10s,oppo_10s
report_root='/home/cxx579/ssw/reports/mtl/aot'
is_dist=false gpu=0 model=resnet evaluation=mtl_1k_ft evaluation.task_name=aot
"""


def send_discord_message(message):
    # Load variables from .env file
    load_dotenv()
    # Retrieve the webhook URL
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        try:
            webhook = DiscordWebhook(url=webhook_url, content=message)
            webhook.execute()
        except Exception as e:
            print(f"🚨 Failed to send discord message: {e}")
    else:
        print("🚨 Webhook URL not found! Make sure .env is set up correctly.")
    

def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    send_discord_message(f"n_samples: {len(X)}")
    send_discord_message("Unique groups:", np.unique(group))
    send_discord_message(f"n_groups: {len(np.unique(group))}")
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=41
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

def train_val_split_sequential(X_train, Y_train, val_ratio=0.125):
    """Splits training data into train and validation sets, ensuring each class is split with the same ratio."""
    unique_classes = np.unique(Y_train)
    X_train_final, X_val = [], []
    Y_train_final, Y_val = [], []

    for cls in unique_classes:
        # Get indices of the current class
        cls_indices = np.where(Y_train == cls)[0]
        split_idx = int(len(cls_indices) * (1 - val_ratio))  # Compute split index for the class

        # Split the data for the current class
        X_train_final.append(X_train[cls_indices[:split_idx]])
        X_val.append(X_train[cls_indices[split_idx:]])
        Y_train_final.append(Y_train[cls_indices[:split_idx]])
        Y_val.append(Y_train[cls_indices[split_idx:]])

    # Concatenate the splits for all classes
    X_train_final = np.concatenate(X_train_final, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_train_final = np.concatenate(Y_train_final, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)

    return X_train_final, X_val, Y_train_final, Y_val


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()


def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)


def evaluate_model(model, data_loader, my_device, loss_fn, cfg):
    model.eval()
    losses = []
    acces = []
    for i, (my_X, my_Y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            logits = model(my_X)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            test_acc = torch.sum(pred_y == true_y)
            test_acc = test_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach().numpy())
            acces.append(test_acc.cpu().detach().numpy())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def setup_data(train_idxs, test_idxs, X_feats, Y, groups, cfg):
    if len(train_idxs) > 0:
        tmp_X_train = X_feats[train_idxs]
        tmp_Y_train = Y[train_idxs]
        group_train = groups[train_idxs]
    else:
        tmp_X_train, tmp_Y_train, group_train = None, None, None

    if len(test_idxs) > 0:
        X_test = X_feats[test_idxs]
        Y_test = Y[test_idxs]
        group_test = groups[test_idxs]
    else:
        X_test, Y_test, group_test = None, None, None
    # when we are not using all the subjects
    if cfg.data.subject_count != -1 and tmp_X_train is not None:
        tmp_X_train, tmp_Y_train, group_train = get_data_with_subject_count(
            cfg.data.subject_count, tmp_X_train, tmp_Y_train, group_train
        )

    # When changing the number of training data, we
    # will keep the test data fixed
    # Train-validation splitting
    if tmp_X_train is not None:
        if cfg.split_method == "held_one_subject_out":
            folds = LeaveOneGroupOut().split(
                tmp_X_train, tmp_Y_train, groups=group_train
            )
            folds = list(folds)
            final_train_idxs, final_val_idxs = folds[0]
            X_train, X_val = (
                tmp_X_train[final_train_idxs],
                tmp_X_train[final_val_idxs],
            )
            Y_train, Y_val = (
                tmp_Y_train[final_train_idxs],
                tmp_Y_train[final_val_idxs],
            )
        elif cfg.split_method == "sequential" or cfg.split_method == "k_shot":
            X_train, X_val, Y_train, Y_val = train_val_split_sequential(tmp_X_train, tmp_Y_train)
        elif cfg.split_method == "random_kfold":
            X_train, X_val, Y_train, Y_val = train_val_split(tmp_X_train, tmp_Y_train, group_train)
    else:
        X_train, X_val, Y_train, Y_val = None, None, None, None


    my_transform = None
    if cfg.augmentation:
        my_transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
    train_dataset = NormalDataset(
        X_train, Y_train, name="train", isLabel=True, transform=my_transform
    ) if X_train is not None else None
    val_dataset = NormalDataset(
        X_val, Y_val, name="val", isLabel=True
        ) if X_val is not None else None
    test_dataset = NormalDataset(
        X_test, Y_test, pid=group_test, name="test", isLabel=True
    )  if X_test is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.evaluation.num_workers,
    ) if train_dataset is not None else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    ) if val_dataset is not None else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    ) if test_dataset is not None else None

    weights = []
    if cfg.data.task_type == "classify" and Y_train is not None:
        weights = get_class_weights(Y_train)
    return train_loader, val_loader, test_loader, weights


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        if len(yhat.size()) == 2:
            yhat = yhat.flatten()
        # return torch.sqrt(self.mse(yhat, y))
        return self.mse(yhat, y)


def train_mlp(model, train_loader, val_loader, cfg, my_device, weights):
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.evaluation.learning_rate, amsgrad=True
    )

    if cfg.data.task_type == "classify":
        if cfg.data.weighted_loss_fn:
            weights = torch.FloatTensor(weights).to(my_device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = RMSELoss()

    early_stopping = EarlyStopping(
        patience=cfg.evaluation.patience, path=cfg.model_path, verbose=True, delta=0.00001
    )
    for epoch in range(cfg.evaluation.num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (my_X, my_Y) in enumerate(train_loader):
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            logits = model(my_X)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach().numpy())
            train_acces.append(train_acc.cpu().detach().numpy())
        val_loss, val_acc = evaluate_model(
            model, val_loader, my_device, loss_fn, cfg
        )

        epoch_len = len(str(cfg.evaluation.num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{cfg.evaluation.num_epoch:>{epoch_len}}] "
            + f"train_loss: {np.mean(train_losses):.5f} "
            + f"valid_loss: {val_loss:.5f}"
        )
        early_stopping(val_loss, model)
        print(print_msg)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model


def mlp_predict(model, data_loader, my_device, cfg):
    predictions_list = []
    true_list = []
    pid_list = []
    probs_list = []  # Added list for probabilities
    model.eval()
    for i, (my_X, my_Y, my_PID) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
                pred_y = model(my_X)
                probs = None  # No probabilities for regression
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)
                logits = model(my_X)
                probs = F.softmax(logits, dim=1)  # Compute class probabilities
                pred_y = torch.argmax(probs, dim=1)  # Get predicted class
                # pred_y = torch.argmax(logits, dim=1)

            true_list.append(true_y.cpu())
            predictions_list.append(pred_y.cpu())
            pid_list.extend(my_PID)
            probs_list.append(probs.cpu())  # Store probabilities
    # Convert lists to tensors
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)
    probs_list = torch.cat(probs_list)  # Concatenate probability tensors
    return (
        torch.flatten(true_list).numpy(),
        torch.flatten(predictions_list).numpy(),
        np.array(pid_list),
        probs_list.numpy(),  # Return probabilities as NumPy array
    )


def init_model(cfg, my_device):
    if cfg.model.is_ae:
        model = EncoderMLP(cfg.data.output_size)
    elif cfg.model.resnet_version > 0:
        model = Resnet(
            output_size=cfg.data.output_size,
            is_eva=True,
            resnet_version=cfg.model.resnet_version,
            epoch_len=cfg.dataloader.epoch_len,
        )
    else:
        model = SSLNET(
            output_size=cfg.data.output_size, flatten_size=1024
        )  # VGG

    if cfg.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    print(model)
    model.to(my_device, dtype=torch.float)
    return model


def setup_model(cfg, my_device):
    model = init_model(cfg, my_device)

    if cfg.evaluation.load_weights:
        print("Loading weights from %s" % cfg.evaluation.flip_net_path)
        load_weights(cfg.evaluation.flip_net_path, model, my_device)
    if cfg.evaluation.freeze_weight:
        freeze_weights(model)
    return model


def get_train_test_split(cfg, X_feats, y, groups):
    # support leave one subject out and split by proportion
    send_discord_message(f"Split method: {cfg.split_method}")
    if cfg.split_method == "held_one_subject_out":
        folds = LeaveOneGroupOut().split(X_feats, y, groups=groups)
    elif cfg.split_method == "random_kfold":
        # Train-test multiple times with a 80/20 random split each
        folds = GroupShuffleSplit(
            cfg.num_split, test_size=0.2, random_state=42
        ).split(X_feats, y, groups=groups)
    elif cfg.split_method == "sequential" or cfg.split_method == "k_shot":
        folds = get_train_test_split_personalized(cfg, X_feats, y, groups)
        send_discord_message(f"Personalized split: {len(folds)} participants")
    else:
        raise ValueError("Invalid split method")
    return folds


def get_train_test_split_personalized(cfg, X_feats, y, groups):
    folds = []
    
    unique_participants = np.unique(groups)
    train_ratio = cfg.train_ratio  # Fraction for training in sequential split

    for participant in unique_participants:
        participant_indices = np.where(groups == participant)[0]
        X_participant, y_participant = X_feats[participant_indices], y[participant_indices]

        unique_activities = np.unique(y_participant)
        train_indices, test_indices = [], []

        if cfg.split_method == "sequential":
            # Configurable sequential split per activity
            for activity in unique_activities:
                activity_indices = np.where(y_participant == activity)[0]
                split_point = int(len(activity_indices) * train_ratio)
                
                train_indices.extend(participant_indices[activity_indices[:split_point]])
                test_indices.extend(participant_indices[activity_indices[split_point:]])

        elif cfg.split_method == "k_shot":
            # K-shot learning: first K samples for training, rest for testing
            K = cfg.k_shot  # Number of shots
            for activity in unique_activities:
                activity_indices = np.where(y_participant == activity)[0]
                
                if len(activity_indices) > K:
                    train_indices.extend(participant_indices[activity_indices[:K]])
                    test_indices.extend(participant_indices[activity_indices[K:]])
                else:
                    train_indices.extend(participant_indices[activity_indices])  # Use all if less than K
         
        folds.append((np.array(train_indices), np.array(test_indices)))
    
    return folds

def train_test_mlp(
    train_idxs,
    test_idxs,
    X_feats,
    y,
    groups,
    cfg,
    my_device,
    log_dir,
    labels=None,
    encoder=None,
):
    model = setup_model(cfg, my_device)
    if cfg.is_verbose:
        print(model)
        summary(model, (3, cfg.evaluation.input_size))
    train_loader, val_loader, test_loader, weights = setup_data(
        train_idxs, test_idxs, X_feats, y, groups, cfg
    )
    # send_discord_message(f"Training MLP on {len(train_idxs)} samples")
    send_discord_message(f"training label distribution: {np.unique(y[train_idxs], return_counts=True)}")
    # send_discord_message(f"labelb weights: {weights}")
    train_mlp(model, train_loader, val_loader, cfg, my_device, weights)
    send_discord_message(f"Training complete. Evaluating on {len(test_idxs)} samples")

    model = init_model(cfg, my_device)

    model.load_state_dict(torch.load(cfg.model_path))
    send_discord_message(f"Model loaded from {cfg.model_path}")
    send_discord_message(f"Model evaluation started")
    y_test, y_test_pred, pid_test, probs = mlp_predict(
        model, test_loader, my_device, cfg
    )
    send_discord_message(f"Model evaluation complete")
    results = classification_scores(y_test, y_test_pred, pid_test, probs, save=True, save_path=log_dir)
    send_discord_message(f"Predictions on test set saved to {log_dir}")
    # save this for every single subject
    # my_pids = np.unique(pid_test)
    # results = []
    # for current_pid in my_pids:
    #     subject_filter = current_pid == pid_test
    #     subject_true = y_test[subject_filter]
    #     subject_pred = y_test_pred[subject_filter]
        
    #     # log_dir = log_dir + f"{str(current_pid)}.csv"
    #     # Make sure the parent directory exists, not the file itself
    #     pathlib.Path(os.path.dirname(log_dir)).mkdir(parents=True, exist_ok=True)
    #     print(f"Final log_dir: {log_dir}")
    #     print(f"Directory exists? {os.path.isdir(os.path.dirname(log_dir))}")
    #     print(f"File exists? {os.path.isfile(f'{log_dir}{str(current_pid)}.csv')}")
    #     # result = classification_scores(subject_true, subject_pred, save=True, save_path=os.path.join(cfg.report_root, f"{str(current_pid)}.csv"))
    #     result = classification_scores(subject_true, subject_pred, save=True, save_path=os.path.join(log_dir, f"{str(current_pid)}.csv"))
    #     results.append(result)
    return results


def evaluate_mlp(X_feats, y, cfg, my_device, logger, log_dir, groups=None):
    """Train a MLP with X_feats and Y.
    Report a variety of performance metrics based on multiple runs."""

    le = None
    labels = None
    if cfg.data.task_type == "classify":
        le = preprocessing.LabelEncoder()
        labels = np.unique(y)
        le.fit(y)
        y = le.transform(y)
    else:
        y = y * 1.0
    # Create a mapping and ensure keys and values are standard Python types
    label_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    print(label_mapping)  # {'Cycling': 0, 'Running': 1, 'Walking': 2}
    send_discord_message(f"Label mapping: {label_mapping}")
    with open(os.path.join(log_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)

    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    folds = get_train_test_split(cfg, X_feats, y, groups)
    total_folds = len(folds)
    results = []
    for fold_num, (train_idxs, test_idxs) in enumerate(folds, 1):
        print(f"Processing fold {fold_num}/{total_folds}")
        send_discord_message(f"Processing fold {fold_num}/{total_folds}")
        print(f"Training on {len(train_idxs)} samples")
        print(f"Evaluating on {len(test_idxs)} samples")
        print(f"Unique labels in training set:{np.unique(y[train_idxs])}")
        try:
            result = train_test_mlp(
                train_idxs,
                test_idxs,
                X_feats,
                y,
                groups,
                cfg,
                my_device,
                os.path.join(log_dir, f"Fold{str(fold_num)}.csv"),
                labels=labels,
                encoder=le,
            )
        except Exception as e:
            error_message = f"🚨 Error in training: {str(e)}"
            send_discord_message(error_message)
        
        results.append(result)
    print(results)
    # Create report directory and generate classification report
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_filename = os.path.basename(cfg.report_path)
    log_path = os.path.join(log_dir, report_filename)
    classification_report(results, log_path)
    


def train_test_rf(
    train_idxs, test_idxs, X_feats, Y, cfg, groups, log_dir, task_type="classify"
):
    X_train, X_test = X_feats[train_idxs], X_feats[test_idxs]
    Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    group_train, group_test = groups[train_idxs], groups[test_idxs]

    # when we are not using all the subjects
    if cfg.data.subject_count != -1:
        X_train, Y_train, group_train = get_data_with_subject_count(
            cfg.data.subject_count, X_train, Y_train, group_train
        )
    if task_type == "classify":
        model = BalancedRandomForestClassifier(
            n_estimators=3000,
            replacement=True,
            sampling_strategy="not minority",
            n_jobs=1,
            random_state=42,
        )
    elif task_type == "regress":
        model = RandomForestRegressor(
            n_estimators=200,  # more is too expensive
            n_jobs=1,
            random_state=42,
            max_features=0.333,
        )

    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    results = []
    for current_pid in np.unique(group_test):
        subject_filter = group_test == current_pid
        subject_true = Y_test[subject_filter]
        subject_pred = Y_test_pred[subject_filter]
        log_dir = log_dir + str(current_pid)+".csv"
        # Make sure the parent directory exists, not the file itself
        pathlib.Path(os.path.dirname(log_dir)).mkdir(parents=True, exist_ok=True)
        result = classification_scores(subject_true, subject_pred,current_pid, None, save = True, save_path=log_dir)
        results.append(result)

    return results

def evaluate_harnet(
    train_idxs,
    test_idxs,
    X_feats,
    y,
    groups,
    cfg,
    my_device,
    log_dir,
    repo='OxWearables/ssl-wearables',
    labels=None,
    encoder=None,
):
    """
    Evaluate HARNet model on test data without training
    
    Args:
        test_idxs: Indices for test data
        X_feats: Input features
        y: Target labels
        groups: Group identifiers for the samples
        cfg: Configuration object
        my_device: Device to run the model on (cuda/cpu)
        repo: Repository path for loading HARNet
        labels: Optional label mapping
        encoder: Optional encoder for labels
    
    Returns:
        List of classification results for each subject
    """
    # Load pre-trained HARNet model
    model = torch.hub.load(repo, 'harnet10', class_num=cfg.data.output_size, pretrained=True)
    model = model.to(my_device)
    model.eval()

    # Setup test data loader
    _, _, test_loader, _ = setup_data(
        train_idxs, test_idxs, X_feats, y, groups, cfg
    )

    # Predict on test data
    y_test, y_test_pred, pid_test, probs = mlp_predict(
        model, test_loader, my_device, cfg
    )
    # Calculate results for each subject
    my_pids = np.unique(pid_test)
    results = []
    for current_pid in my_pids:
        subject_filter = current_pid == pid_test
        subject_true = y_test[subject_filter]
        subject_pred = y_test_pred[subject_filter]
        log_dir = log_dir + str(current_pid)+".csv"
        # Make sure the parent directory exists, not the file itself
        pathlib.Path(os.path.dirname(log_dir)).mkdir(parents=True, exist_ok=True)
        result = classification_scores(subject_true, subject_pred, pid_test, probs, save=True, save_path=log_dir)
        results.append(result)
    
    return results
def evaluate_harnet_classification(X_feats, y, cfg, my_device, logger, log_dir, groups=None, repo='OxWearables/ssl-wearables'):
    """Evaluate HARNet model on the given data and generate classification reports.
    Reports a variety of performance metrics based on multiple runs.
    
    Args:
        X_feats: Input features
        y: Target labels
        cfg: Configuration object
        my_device: Device to run model on (cuda/cpu)
        logger: Logger object for recording progress
        groups: Optional group identifiers for the samples
        repo: Repository path for loading HARNet
    """
    # Handle label encoding for classification
    le = None
    labels = None
    if cfg.data.task_type == "classify":
        le = preprocessing.LabelEncoder()
        labels = np.unique(y)
        le.fit(y)
        y = le.transform(y)
    else:
        y = y * 1.0

    # Convert DataFrame to numpy if needed
    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    # Ensure input data has correct shape for HARNet (3 channels)
    if len(X_feats.shape) == 2:
        # Reshape 2D data to 3D (batch_size, 3, sequence_length)
        seq_length = X_feats.shape[1]
        X_feats = X_feats.reshape(-1, 3, seq_length // 3)
    
        # Get cross-validation folds and collect results
    folds = get_train_test_split(cfg, X_feats, y, groups)
    results = []
    
    for train_idxs, test_idxs in folds:
        result = evaluate_harnet(
            train_idxs,
            test_idxs,
            X_feats,
            y,
            groups,
            cfg,
            my_device,
            log_dir,
            repo=repo,
            labels=labels,
            encoder=le,
        )
        results.extend(result)

    # Create report directory and generate classification report
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_filename = os.path.basename(cfg.report_path)
    log_dir = os.path.join(log_dir, report_filename)
    classification_report(results, log_dir)

    return results

def evaluate_feats(X_feats, Y, cfg, logger, groups=None, task_type="classify"):
    """Train a random forest with X_feats and Y.
    Report a variety of performance metrics based on multiple runs."""

    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    # Train-test multiple times with a 80/20 random split each
    # Five-fold or Held one subject out
    folds = get_train_test_split(cfg, X_feats, Y, groups)
    print("loading done")
    results = Parallel(n_jobs=1)(
        delayed(train_test_rf)(
            train_idxs, test_idxs, X_feats, Y, cfg, groups, task_type
        )
        for train_idxs, test_idxs in folds
    )
    results = np.array(results)

    results = np.array(
        [
            fold_result
            for fold_results in results
            for fold_result in fold_results
        ]
    )

    print(results)
    pathlib.Path(cfg.report_root).mkdir(parents=True, exist_ok=True)
    # report_filename = os.path.basename(cfg.report_path)
    # log_dir = os.path.join(log_dir, report_filename)
    classification_report(results, cfg.report_path)

def train_and_save_model(train_loader, val_loader, cfg, my_device, weights):
    model = setup_model(cfg, my_device)
    train_mlp(model, train_loader, val_loader, cfg, my_device, weights)
    # torch.save(model.state_dict(), model_path)
    # return model_path
def evaluate_saved_model(test_loader, cfg, my_device, log_dir):
    model = init_model(cfg, my_device)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    y_test, y_test_pred, pid_test, probs = mlp_predict(model, test_loader, my_device, cfg)

    my_pids = np.unique(pid_test)
    results = []
    for current_pid in my_pids:
        subject_filter = current_pid == pid_test
        subject_true = y_test[subject_filter]
        subject_pred = y_test_pred[subject_filter]
        subject_probs = probs[subject_filter]  # Filter probabilities for the current subject
        subject_log_dir = os.path.join(log_dir, f"{str(current_pid)}.csv")
        os.makedirs(os.path.dirname(subject_log_dir), exist_ok=True)
        result = classification_scores(subject_true, subject_pred, [current_pid] * len(subject_true), subject_probs, save=True, save_path=subject_log_dir)
        results.append(result)
       
    return results

def cross_dataset_evaluation(train_data, test_data, cfg, my_device, log_dir):
    # Load dataset1 (train_data)
    X_train = train_data[0]
    Y_train = train_data[1]
    P_train = train_data[2]

    # Load dataset2 (test_data)
    X_test = test_data[0]
    Y_test = test_data[1]
    P_test = test_data[2]

    # Label encoding for dataset1
    le_train = preprocessing.LabelEncoder()
    le_train.fit(Y_train)
    Y_train = le_train.transform(Y_train)

    # Label encoding for dataset2
    le_test = preprocessing.LabelEncoder()
    le_test.fit(Y_test)
    Y_test = le_test.transform(Y_test)

    # Create label mapping between dataset1 and dataset2
    label_mapping = {str(k): int(v) for k, v in zip(le_train.classes_, le_test.transform(le_train.classes_))}
    with open(os.path.join(log_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)

    # Setup data loaders for training and validation
    train_loader, val_loader, _, weights = setup_data(
        np.arange(len(X_train)), np.array([]), X_train, Y_train, P_train, cfg
    )
    _, _, test_loader, _ = setup_data(
        np.array([]), np.arange(len(X_test)), X_test, Y_test, P_test, cfg
    )

    # Train the model on dataset1 and save it
    train_and_save_model(train_loader, val_loader, cfg, my_device, weights)

    # Evaluate the saved model on dataset2
    results = evaluate_saved_model(test_loader, cfg, my_device, label_mapping, log_dir)
    print(results)
    # Create report directory and generate classification report
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_filename = os.path.basename(cfg.report_path)
    log_path = os.path.join(log_dir, report_filename)
    classification_report(results, log_path)
def map_labels(Y_test,cfg, log_dir, reference_method, ignore_cross_dataset_mapping = False):
    if ignore_cross_dataset_mapping:
        le_test = preprocessing.LabelEncoder()
        le_test.fit(Y_test)
        Y_test = le_test.transform(Y_test)
        # Create a mapping and ensure keys and values are standard Python types
        label_mapping = {str(k): int(v) for k, v in zip(le_test.classes_, le_test.transform(le_test.classes_))}
        print(f"Final label mapping: {label_mapping}")  # {'Cycling': 0, 'Running': 1, 'Walking': 2}
        with open(os.path.join(log_dir, "eval_label_mapping.json"), "w") as f:
            json.dump(label_mapping, f)
        return Y_test
    if reference_method:
        dataset1 = cfg.data.dataset_name
        dataset2 = cfg.evaluation_data.dataset_name
        mapping_path = os.path.join(get_original_cwd(), cfg.mapping_path)
        mapper = DatasetLabelMapper(mapping_path)
        Y_test = mapper.map_labels_to_numeric(Y_test, dataset2)
        label_mapping = mapper.get_label_mapping(dataset2)
        print(f"Final label mapping: {label_mapping}")  # {'Cycling': 0, 'Running': 1, 'Walking': 2}
        with open(os.path.join(log_dir, "eval_label_mapping.json"), "w") as f:
            json.dump(label_mapping, f)
    else:
        if hasattr(cfg, 'cross_dataset_mapping') and cfg.cross_dataset_mapping:
            # Load mapping from yaml/dict config
            cross_dataset_map = cfg.cross_dataset_mapping
            print(f"Applying cross-dataset mapping: {cross_dataset_map}")
            
            # Create a new array with mapped labels
            Y_test_mapped = np.array([cross_dataset_map.get(label, label) for label in Y_test])
            print(f"Original labels: {np.unique(Y_test)}")
            print(f"Mapped labels: {np.unique(Y_test_mapped)}")
            
            # Use the mapped labels for encoding
            Y_test = Y_test_mapped
        if hasattr(cfg, 'trained_label_mapping_path') and cfg.trained_label_mapping_path:
            # expand ~ to the user's home directory
            cfg.trained_label_mapping_path = os.path.expanduser(cfg.trained_label_mapping_path)
            # Load label mapping from file
            with open(cfg.trained_label_mapping_path, "r") as f:
                label_mapping = json.load(f)
                print(f"Loaded label mapping from {cfg.trained_label_mapping_path}")
                # map Y_test labels to numeric labels using the loaded mapping
                Y_test = np.array([label_mapping.get(label, label) for label in Y_test])
        else:
            le_test = preprocessing.LabelEncoder()
            le_test.fit(Y_test)
            Y_test = le_test.transform(Y_test)
            # Create a mapping and ensure keys and values are standard Python types
            label_mapping = {str(k): int(v) for k, v in zip(le_test.classes_, le_test.transform(le_test.classes_))}
            print(f"Final label mapping: {label_mapping}")  # {'Cycling': 0, 'Running': 1, 'Walking': 2}
            with open(os.path.join(log_dir, "eval_label_mapping.json"), "w") as f:
                json.dump(label_mapping, f)
    return Y_test
def evaluate_pretrained_model(test_data, cfg, my_device, log_dir):
    X_test = test_data[0]
    Y_test = test_data[1]
    P_test = test_data[2]
   
    Y_test = map_labels(Y_test, cfg, log_dir, reference_method = False, ignore_cross_dataset_mapping = True)
    
    _, _, test_loader, _ = setup_data(
        np.array([]), np.arange(len(X_test)), X_test, Y_test, P_test, cfg
    )
    
    # Evaluate the saved model on dataset2
    results = evaluate_saved_model(test_loader, cfg, my_device, log_dir)
    print(results)
    # Create report directory and generate classification report
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_filename = os.path.basename(cfg.report_path)
    log_path = os.path.join(log_dir, report_filename)
    classification_report(results, log_path)


    """Our baseline handcrafted features. xyz is a window of shape (N,3)"""
def handcraft_features(xyz, sample_rate):

    feats = {}
    feats["xMean"], feats["yMean"], feats["zMean"] = np.mean(xyz, axis=0)
    feats["xStd"], feats["yStd"], feats["zStd"] = np.std(xyz, axis=0)
    feats["xRange"], feats["yRange"], feats["zRange"] = np.ptp(xyz, axis=0)

    x, y, z = xyz.T

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["xyCorr"] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats["yzCorr"] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats["zxCorr"] = np.nan_to_num(np.corrcoef(z, x)[0, 1])

    m = np.linalg.norm(xyz, axis=1)

    feats["mean"] = np.mean(m)
    feats["std"] = np.std(m)
    feats["range"] = np.ptp(m)
    feats["mad"] = stats.median_abs_deviation(m)
    if feats["std"] > 0.01:
        feats["skew"] = np.nan_to_num(stats.skew(m))
        feats["kurt"] = np.nan_to_num(stats.kurtosis(m))
    else:
        feats["skew"] = feats["kurt"] = 0
    feats["enmomean"] = np.mean(np.abs(m - 1))

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend=False,
        average="median",
    )

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["pentropy"] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to find dominant freqs
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend="constant",
        average="median",
    )

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats["f1"] = peak_freqs[peak_ranks[0]]
        feats["f2"] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats["f1"] = feats["f2"] = peak_freqs[peak_ranks[0]]
    else:
        feats["f1"] = feats["f2"] = 0

    return feats


def forward_by_batches(cnn, X, cnn_input_size, my_device="cpu"):
    """Forward pass model on a dataset. Includes resizing to model input size.
    Do this by batches so that we don't blow up the memory.
    """

    BATCH_SIZE = 1024

    X_feats = []
    cnn.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(X), BATCH_SIZE)):
            batch_end = i + BATCH_SIZE
            X_batch = X[i:batch_end]

            # Resize to expected input length
            X_batch = resize(X_batch, length=cnn_input_size)
            X_batch = X_batch.astype("f4")  # PyTorch defaults to float32
            X_batch = np.transpose(
                X_batch, (0, 2, 1)
            )  # channels first: (N,M,3) -> (N,3,M) channel first format
            X_batch = torch.from_numpy(X_batch)
            X_batch = X_batch.to(my_device, dtype=torch.float)

            if my_device == "cpu":
                X_feats.append(cnn(X_batch).numpy())
            else:
                X_feats.append(cnn(X_batch).cpu().numpy())

    X_feats = np.concatenate(X_feats)

    return X_feats


def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )

    return X


def get_data_with_subject_count(subject_count, X, y, pid):
    subject_list = np.unique(pid)

    if subject_count == len(subject_list):
        valid_subjects = subject_list
    else:
        valid_subjects = subject_list[:subject_count]

    pid_filter = [my_subject in valid_subjects for my_subject in pid]

    filter_X = X[pid_filter]
    filter_y = y[pid_filter]
    filter_pid = pid[pid_filter]
    return filter_X, filter_y, filter_pid


def load_weights(weight_path, model, my_device):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split(".")[
        0
    ]  # get head of first key
    if head == "module":
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {
            k.partition("module.")[2]: pretrained_dict_v2[k]
            for k in pretrained_dict_v2.keys()
        }

    if hasattr(model, "module"):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))
def downsample_data(X, input_size):
    # Original X shape: (1861541, 1000, 3) for capture24
    print("Original X shape:", X.shape)

    if X.shape[1] == input_size:
        print("No need to downsample")
        X_downsampled = X
    else:
        X_downsampled = resize(X, input_size)
    X_downsampled = X_downsampled.astype(
        "f4"
    )  # PyTorch defaults to float32
    # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
    X_downsampled = np.transpose(X_downsampled, (0, 2, 1))
    print("X transformed shape:", X_downsampled.shape)
    return X_downsampled

@hydra.main(config_path="conf", config_name="config_eva_person")
def main(cfg):
    """Evaluate hand-crafted vs deep-learned features"""

    logger = logging.getLogger(cfg.evaluation.evaluation_name)
    logger.setLevel(logging.INFO)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = os.path.join(
        get_original_cwd(),
        cfg.evaluation.evaluation_name + "_" + dt_string + ".log",
    )
    dtm_string = now.strftime("%Y-%m-%d_%H-%M")
    # log_dir_r = os.path.join(cfg.report_root, dtm_string)
    # os.makedirs(log_dir_r, exist_ok=True)
    log_dir_r = pathlib.Path(os.path.expanduser(cfg.report_root)) / dtm_string
    log_dir_r.mkdir(parents=True, exist_ok=True)
    if cfg.use_pretrained == False:
        cfg.model_path = os.path.join(get_original_cwd(), dt_string + "tmp.pt")
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(str(OmegaConf.to_yaml(cfg)))
    send_discord_message("Downstream task started!")
    
    # For reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    print(cfg.report_path)
    # ----------------------------
    #
    #            Main
    #
    # ----------------------------

    # Load dataset
    X = np.load(cfg.data.X_path)
    Y = np.load(cfg.data.Y_path, allow_pickle=True)
    P = np.load(cfg.data.PID_path, allow_pickle=True)  # participant IDs

    sample_rate = cfg.data.sample_rate
    task_type = cfg.data.task_type
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"
    print("Device:", my_device)
    # Expected shape of downstream X and Y
    # X: T x (Sample Rate*Epoch len) x 3
    # Y: T,
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    if task_type == "classify":
        print("\nLabel distribution:")
        print(pd.Series(Y).value_counts())
    elif task_type == "regress":
        print("\nOutput distribution:")
        Y_qnt = pd.Series(Y).quantile((0, 0.25, 0.5, 0.75, 1))
        Y_qnt.index = ("min", "25th", "median", "75th", "max")
        print(Y_qnt)

    if cfg.use_pretrained == False:

        if cfg.evaluation.feat_hand_crafted:
            task_message ="""\n
            ##############################################
                        Hand-crafted features+RF
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            # Extract hand-crafted features
            print("Extracting features...")
            X_handfeats = pd.DataFrame(
                [handcraft_features(x, sample_rate=sample_rate) for x in tqdm(X)]
            )
            print("X_handfeats shape:", X_handfeats.shape)

            print("Train-test RF...")
            evaluate_feats(
                X_handfeats, Y, cfg, logger, groups=P, task_type=task_type
            )

        if cfg.evaluation.feat_random_cnn:
            task_message = """\n
            ##############################################
                        Random CNN features+RF
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            # Extract CNN features
            print("Extracting features...")
            if cfg.evaluation.network == "vgg":
                model = cnn1()
            else:
                # get cnn
                model = Resnet(output_size=cfg.data.output_size, cfg=cfg)
            model.to(my_device, dtype=torch.float)
            input_size = cfg.evaluation.input_size

            X_deepfeats = forward_by_batches(model, X, input_size, my_device)
            print("X_deepfeats shape:", X_deepfeats.shape)

            print("Train-test RF...")
            evaluate_feats(X_deepfeats, Y, cfg, logger, groups=P)

        if cfg.evaluation.flip_net:
            task_message ="""\n
            ##############################################
                        Flip_net+RF
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            # Extract CNN features
            print("Extracting features...")
            cnn = cnn1()
            cnn.to(my_device, dtype=torch.float)
            load_weights(cfg.evaluation.flip_net_path, cnn, my_device)
            input_size = cfg.evaluation.input_size

            X_deepfeats = forward_by_batches(cnn, X, input_size, my_device)
            print("X_deepfeats shape:", X_deepfeats.shape)

            print("Train-test RF...")
            evaluate_feats(X_deepfeats, Y, cfg, logger, groups=P)

        """
        Start of MLP classifier evaluation
        """

        if cfg.evaluation.flip_net_ft:
            task_message = """\n
            ##############################################
                        Flip_net+MLP
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            X_downsampled = downsample_data(X, cfg.evaluation.input_size)

            print("Train-test Flip_net+MLP...")
            evaluate_mlp(X_downsampled, Y, cfg, my_device, logger,log_dir_r, groups=P)
        if cfg.evaluation.harnet_ft:
            task_message = """\n
            ##############################################
                        HARNET pretrained
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            X_downsampled = downsample_data(X, cfg.evaluation.input_size)

            print("Evaluate HARNET...")
            evaluate_harnet_classification(X_downsampled, Y, cfg, my_device, logger, log_dir_r, groups=P)
    else:
        if cfg.cross_dataset:
            task_message ="""\n
            ##############################################
                        Cross-dataset evaluation
            ##############################################
            """
            print(task_message)
            send_discord_message(task_message)
            
            # Load evaluation dataset
            X_eval = np.load(cfg.evaluation_data.X_path)
            Y_eval = np.load(cfg.evaluation_data.Y_path)
            P_eval = np.load(cfg.evaluation_data.PID_path)  # participant IDs
            X_downsampled = downsample_data(X, cfg.evaluation.input_size)
            X_eval_downsampled = downsample_data(X_eval, cfg.evaluation.input_size)
            test_data = (X_eval_downsampled, Y_eval, P_eval)
            train_data = (X_downsampled, Y, P)
            cross_dataset_evaluation(train_data, test_data, cfg, my_device, log_dir_r)
            
        else:
            task_message = """\n
                ##############################################
                            Load a pretrained model
                ##############################################
                """
            print(task_message)
            send_discord_message(task_message)
            # Load evaluation dataset
            X_eval = np.load(cfg.evaluation_data.X_path)
            Y_eval = np.load(cfg.evaluation_data.Y_path, allow_pickle=True)
            P_eval = np.load(cfg.evaluation_data.PID_path, allow_pickle=True)  # participant IDs
            print("X_eval shape:", X_eval.shape)
            print("Y_eval shape:", Y_eval.shape)
            print("P_eval shape:", P_eval.shape)
            print("\nLabel distribution:")
            print(pd.Series(Y_eval).value_counts())
            X_eval_downsampled = downsample_data(X_eval, cfg.evaluation.input_size)
            print("labels:", np.unique(Y_eval))
            test_data = (X_eval_downsampled, Y_eval, P_eval)
            evaluate_pretrained_model(test_data, cfg, my_device, log_dir_r)
           
if __name__ == "__main__":
    main()
