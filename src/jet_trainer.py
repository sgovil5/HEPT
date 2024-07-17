import nni
import sys
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch_geometric
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from utils import set_seed, get_optimizer, log, get_lr_scheduler, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from sklearn.metrics import accuracy_score

def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    output = model(data)
    output = torch_geometric.nn.global_mean_pool(output, data.batch)
    y = data.y.float().unsqueeze(0)
    
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), output.detach()

@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    output = model(data)
    output = torch_geometric.nn.global_mean_pool(output, data.batch)
    y = data.y.float().unsqueeze(0)
    loss = criterion(output, y)
    return loss.item(), output.detach()

def run_one_epoch(model, optimizer, criterion, data_loader, phase, epoch, device, metrics, lr_s):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase
    pbar = tqdm(data_loader, disable=__name__ != "__main__")
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for idx, data in enumerate(pbar):
        data = data.to(device)
        batch_loss, batch_output = run_one_batch(model, optimizer, criterion, data, lr_s)
        
        # Calculate accuracy
        predicted = torch.argmax(batch_output, dim=1)
        actual = torch.argmax(data.y, dim=0)
        correct = (predicted == actual).item()
        
        total_loss += batch_loss
        total_correct += correct
        total_samples += 1
        
        batch_acc = correct / 1
        
        metrics["loss"].update(batch_loss)
        metrics["accuracy"].update(batch_acc)

        desc = f"[Epoch {epoch}] {phase}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}"
        if idx == len(data_loader) - 1:
            avg_loss = total_loss / len(data_loader)
            avg_acc = total_correct / total_samples
            desc = f"[Epoch {epoch}] {phase}, loss: {avg_loss:.4f}, acc: {avg_acc:.4f}"
            reset_metrics(metrics)
        pbar.set_description(desc)
    
    metric_res = {
        "loss": total_loss / len(data_loader),
        "accuracy": total_correct / total_samples,
    }
    return metric_res

def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()

def compute_metrics(metrics):
    return {f"{name}": metrics[f"{name}"].compute().item() for name in ["accuracy"]} | {
        "loss": metrics["loss"].compute().item()
    }

def update_metrics(metrics, labels, outputs):
    pred = outputs.argmax(dim=1).cpu()
    labels = labels.cpu()

    acc = accuracy_score(labels, pred)

    metrics["accuracy"].update(acc)
    return acc

def run_one_seed(config, tune=False):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name
    log(f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, Note: {config['note']}")

    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    log_dir = dataset_dir / "logs" / f"{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}"
    log(f"Log dir: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir) if config["log_tensorboard"] else None

    set_seed(config["seed"])
    dataset = get_dataset(dataset_name, dataset_dir)
    loaders = get_data_loader(dataset, dataset.idx_split, batch_size=config["batch_size"])

    model_kwargs = config['model_kwargs']
    # model_kwargs['num_classes'] = config['model_kwargs']['num_classes']  # Ensure this line exists
    model = get_model(model_name, model_kwargs, dataset)
    if config.get("resume", False):
        log(f"Resume from {config['resume']}")
        model_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * len(loaders["train"])
    lr_s = get_lr_scheduler(opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"])
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])

    main_metric = config["main_metric"]
    metric_names = ["accuracy"]
    metrics = {f"{name}": MeanMetric() for name in metric_names}
    metrics["loss"] = MeanMetric()

    coef = 1 if config["mode"] == "max" else -1
    best_epoch, best_train = 0, {metric: -coef * float("inf") for metric in metrics.keys()}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "accuracy": ["Multiline", ["train/accuracy", "valid/accuracy", "test/accuracy"]],
            }
        }
        writer.add_custom_scalars(layout)

    for epoch in range(config["num_epochs"]):
        train_res = run_one_epoch(model, opt, criterion, loaders["train"], "train", epoch, device, metrics, lr_s)
        valid_res = run_one_epoch(model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
        test_res = run_one_epoch(model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

        if lr_s is not None and isinstance(lr_s, ReduceLROnPlateau):
            lr_s.step(valid_res[config["lr_scheduler_metric"]])

        if (valid_res[main_metric] * coef) > (best_valid[main_metric] * coef):
            best_epoch, best_train, best_valid, best_test = epoch, train_res, valid_res, test_res
            torch.save(model.state_dict(), log_dir / "best_model.pt")

        print(
            f"[Epoch {epoch}] Best epoch: {best_epoch}, train: {best_train[main_metric]:.4f}, "
            f"valid: {best_valid[main_metric]:.4f}, test: {best_test[main_metric]:.4f}"
        )
        print("=" * 50), print("=" * 50)

        if writer is not None:
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            for phase, res in zip(["train", "valid", "test"], [train_res, valid_res, test_res]):
                for k, v in res.items():
                    writer.add_scalar(f"{phase}/{k}", v, epoch)
            for phase, res in zip(["train", "valid", "test"], [best_train, best_valid, best_test]):
                for k, v in res.items():
                    writer.add_scalar(f"best_{phase}/{k}", v, epoch)

def main():
    parser = argparse.ArgumentParser(description="Train a model for jet classification.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/jetclass/jetclass_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/jetclass/jetclass_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    run_one_seed(config)

if __name__ == "__main__":
    main()