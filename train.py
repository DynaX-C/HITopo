import os
import torch
import csv
import json
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from model import build_model
from model.variants import apply_variant
from utils.data_utils import set_seed, split_graphs_by_pdbid, load_graph_dict
from utils.config import parse_config_and_override
from data.dataloader import data_loader
import warnings
warnings.filterwarnings("ignore")

def train_and_validate(model, 
                       train_atom_loader, train_nb_atom_loader, train_bd_atom_loader, train_bond_loader, train_angle_loader, train_dihedral_loader,
                       val_atom_loader, val_nb_atom_loader, val_bd_atom_loader, val_bond_loader, val_angle_loader, val_dihedral_loader,
                       epochs=5000, lr=5e-4, device='cuda', patience=100, save_path='best_model.pt', log_path='log.csv'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float('inf')
    best_epoch = 0
    patience_counter = 0

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_r2", "train_r", "val_mse", "val_r2", "val_r"])

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        train_preds = []
        train_targets = []

        for atom_batch, nb_atom_batch, bd_atom_batch, bond_batch, angle_batch, dihedral_batch in zip(
            train_atom_loader, train_nb_atom_loader, train_bd_atom_loader, train_bond_loader, train_angle_loader, train_dihedral_loader
        ):
            atom_batch, nb_atom_batch, bd_atom_batch = atom_batch.to(device), nb_atom_batch.to(device), bd_atom_batch.to(device)
            bond_batch, angle_batch, dihedral_batch = bond_batch.to(device), angle_batch.to(device), dihedral_batch.to(device)

            batch = {
                'atom_graph': atom_batch,
                'nb_atom_graph': nb_atom_batch,
                'bd_atom_graph': bd_atom_batch,
                'bond_graph': bond_batch,
                'angle_graph': angle_batch,
                'dihedral_graph': dihedral_batch
            }

            labels = nb_atom_batch.y.to(device)
            optimizer.zero_grad()
            output = model(batch)

            train_preds.append(output.detach().cpu())
            train_targets.append(labels.detach().cpu())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / n_train_batches
        train_r2 = r2_score(torch.cat(train_targets).numpy(), torch.cat(train_preds).numpy())
        train_r, _ = pearsonr(torch.cat(train_targets).numpy(), torch.cat(train_preds).numpy())

        # ---------- Validation ----------
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for atom_batch, nb_atom_batch, bd_atom_batch, bond_batch, angle_batch, dihedral_batch in zip(
                val_atom_loader, val_nb_atom_loader, val_bd_atom_loader, val_bond_loader, val_angle_loader, val_dihedral_loader
            ):
                atom_batch, nb_atom_batch, bd_atom_batch = atom_batch.to(device), nb_atom_batch.to(device), bd_atom_batch.to(device)
                bond_batch, angle_batch, dihedral_batch = bond_batch.to(device), angle_batch.to(device), dihedral_batch.to(device)

                batch = {
                    'atom_graph': atom_batch,
                    'nb_atom_graph': nb_atom_batch,
                    'bd_atom_graph': bd_atom_batch,
                    'bond_graph': bond_batch,
                    'angle_graph': angle_batch,
                    'dihedral_graph': dihedral_batch
                }

                labels = nb_atom_batch.y.to(device)
                output = model(batch)

                all_preds.append(output.detach().cpu())
                all_targets.append(labels.detach().cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        val_mse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        r, _ = pearsonr(targets, preds)

        print(f"Epoch {epoch+1}/{epochs} | Train -Loss: {avg_train_loss:.4f} -R²: {train_r2:.4f} -r: {train_r:.4f} | "
              f"Valid -MSE: {val_mse:.4f} -R²: {r2:.4f} -r: {r:.4f}")

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, train_r2, train_r, val_mse, r2, r])

        current_r = r
        current_mse = val_mse
        current_val = val_mse
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model found at epoch {epoch+1} | Valid MSE={current_mse:.4f} r={current_r:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {best_epoch+1} (best MSE = {best_val:.4f})")
                break

if __name__ == "__main__":

    args = parse_config_and_override()
    args = apply_variant(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)

    if args.data_mode in ["pdbbind2020", "cleansplit"]:
        n = args.fold
        split_data = split_graphs_by_pdbid(
            graph_pt_path=args.graph_pt_path,
            split_json_path=args.split_json_template.format(fold=n)
        )
        train_fold = split_data["train"]
        val_fold = split_data["val"]

    elif args.data_mode == "temporalsplit":
        train_fold = load_graph_dict(args.train_graph_pt_path)
        val_fold = load_graph_dict(args.valid_graph_pt_path)

    else:
        raise ValueError(f"Unknown data_mode: {args.data_mode}")

    train_atom_loader, train_nb_atom_loader, train_bd_atom_loader, train_bond_loader, train_angle_loader, train_dihedral_loader = data_loader(
        train_fold["atom"],
        train_fold["nb_atom"],
        train_fold["bd_atom"],
        train_fold["bond"],
        train_fold["angle"],
        train_fold["dihedral"],
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )

    val_atom_loader, val_nb_atom_loader, val_bd_atom_loader, val_bond_loader, val_angle_loader, val_dihedral_loader = data_loader(
        val_fold["atom"],
        val_fold["nb_atom"],
        val_fold["bd_atom"],
        val_fold["bond"],
        val_fold["angle"],
        val_fold["dihedral"],
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
    )

    model = build_model(args)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.data_mode in ["pdbbind2020", "cleansplit"]:
        save_path = os.path.join(out_dir, f"best_{args.variant}_{args.model}_{args.data_mode}_f{n}.pt")
        log_path = os.path.join(out_dir, f"log_{args.variant}_{args.model}_{args.data_mode}_f{n}.csv")
        config_path = os.path.join(out_dir, f"config_{args.variant}_{args.model}_{args.data_mode}_f{n}.json")
    else:
        save_path = os.path.join(out_dir, f"best_{args.variant}_{args.model}_{args.data_mode}_s{args.seed}.pt")
        log_path = os.path.join(out_dir, f"log_{args.variant}_{args.model}_{args.data_mode}_s{args.seed}.csv")
        config_path = os.path.join(out_dir, f"config_{args.variant}_{args.model}_{args.data_mode}_s{args.seed}.json")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    train_and_validate(
        model,
        train_atom_loader, train_nb_atom_loader, train_bd_atom_loader,
        train_bond_loader, train_angle_loader, train_dihedral_loader,
        val_atom_loader, val_nb_atom_loader, val_bd_atom_loader,
        val_bond_loader, val_angle_loader, val_dihedral_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        save_path=save_path,
        log_path=log_path,
    )
