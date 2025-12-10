import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch

plt.style.use(["science"])

figsize = (8, 6)


def plot_roc_curve(fpr, tpr, target_tpr, auroc, save_dir):

    index = torch.argmin((tpr - target_tpr).abs())
    fpr_at_target = fpr[index].cpu().numpy().item()

    fig, ax = plt.subplots(figsize=figsize)
    # self.metrics["roc"].plot(score=True, ax=ax)
    ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), label=f'ROC curve (AUC = {auroc.cpu().item():.2f})')
    ax.plot(
        [fpr_at_target],
        [target_tpr],
        marker="o",
        markersize=5,
        label=f"FPR @99pct TPR: {100*fpr_at_target: .2f}\%",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.grid(True, linestyle="--", linewidth=1)
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(save_path)
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def plot_precision_recall_curve(precision, recall, target_recall, precision_at_target, score_threshold, save_dir):

    fig, ax = plt.subplots(figsize=figsize)
    # self.metrics["precision_recall"].plot(ax=ax)
    ax.plot(recall.cpu().numpy(), precision.cpu().numpy(), label="Precision-Recall curve")
    ax.plot(
        [target_recall],
        [precision_at_target],
        marker="o",
        markersize=5,
        label=f'Precision {precision_at_target * 100 : .2f}\% Recall: {target_recall * 100 :.2f}\%\n Score threshold: {score_threshold:.2f}',
    )
    ax.grid(True, linestyle="--", linewidth=1)
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "precision_recall.png")
    plt.savefig(save_path)
    print(f"Precision-Recall curve saved to: {save_path}")
    plt.close()


def plot_score_histogram(hist_dict, left_edges, widths, save_dir):
    
    fig, ax = plt.subplots(figsize=figsize)
    for label, hist in hist_dict.items():
        ax.bar(
            x=left_edges,
            height=hist.cpu().numpy(),
            width=widths,
            align="edge",
            label=label.item(),
            alpha=0.8
        )
    ax.legend(title="Hit Label")
    ax.set_xlabel("Hit score")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Classification score histogram")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "score_histogram.png")
    plt.savefig(save_path)
    print(f"Score histogram saved to: {save_path}")
    plt.close()


def plot_eff_pur(X, Y, xerr, x_label, y_label, title):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(X, Y, xerr=xerr, elinewidth=1, fmt='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, ax


def plot_hit_metrics(eta_eff, pt_eff, eta_pur, eta_bins, pt_bins, save_dir, metrics):

    eta_eff, pt_eff, eta_pur, eta_bins, pt_bins = (
        t.numpy() for t in [eta_eff, pt_eff, eta_pur, eta_bins, pt_bins]
    )
    etas = (eta_bins[:-1] + eta_bins[1:]) / 2
    pts = (pt_bins[:-1] + pt_bins[1:]) / 2

    eta_err = np.diff(eta_bins) / 2
    pt_err = np.diff(pt_bins) / 2

    fig, ax = plot_eff_pur(etas, eta_eff, eta_err, r"$\eta$", "Efficiency", r"Hit efficiency against $\eta$")
    ax.set_ybound(0.9, 1.1)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "eff_vs_eta.png")
    plt.savefig(save_path)
    print(f"Efficiency vs eta plot saved to: {save_path}")
    plt.close()

    fig, ax = plot_eff_pur(etas, eta_pur, eta_err, r"$\eta$", "Purity", r"Hit purity against $\eta$")
    ax.set_ylim(0.95, 1.05)
    ax.text(-4, 1.04, f"Point cloud reduction factor: {metrics['hit_reduction']*100 : .2f}%")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pur_vs_eta.png")
    plt.savefig(save_path)
    print(f"Purity vs eta plot saved to: {save_path}")
    plt.close()

    fig, ax = plot_eff_pur(pts, pt_eff, pt_err, r"$p_T$", "Efficiency", r"Hit efficiency against $p_T$")
    ax.set_ylim(0.95, 1.05)
    ax.text(0, 1.04, f"Point cloud reduction factor: {metrics['hit_reduction']*100 : .2f}%")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "eff_vs_pt.png")
    plt.savefig(save_path)
    print(f"Efficiency vs pT plot saved to: {save_path}")
    plt.close()



class ResultPlotter(L.pytorch.callbacks.Callback):

    def __init__(self, save_dir, score_threshold: float=0.5, pt_scale_factor:float=1., *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.save_dir = save_dir
        self.score_threshold = score_threshold
        self.pt_scale_factor = pt_scale_factor


    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("Computing metrics...")

        metrics = pl_module.metrics.compute()
        hit_metrics = pl_module.hit_metrics.compute(self.score_threshold, self.pt_scale_factor)

        print("Plotting ROC curve ...")
        # plot ROC curve
        target_tpr = 0.99
        fpr, tpr, threshold = metrics['roc']
        
        plot_roc_curve(fpr, tpr, target_tpr, metrics["auroc"], save_dir)
        
        # plot precision-recall curve
        print("Plotting Precision-Recall curve ...")
        precision, recall, thresholds = (t.cpu() for t in metrics['precision_recall'])
        target_recall = 0.995
        min_idx = torch.argmin((recall - target_recall).abs())
        target_recall = recall[min_idx]
        precision_at_target = precision[min_idx]
        threshold = thresholds[min_idx]
        plot_precision_recall_curve(precision, recall, target_recall, precision_at_target, threshold, save_dir)

        # plot score histogram
        print("Plotting score histogram ...")
        hist_dict, bin_edges = (
            metrics["score_histogram"],
            metrics["bin_edges"],
        )
        bin_edges = bin_edges.cpu().numpy()
        left_edges = bin_edges[:-1]
        widths = np.diff(bin_edges)
        plot_score_histogram(hist_dict, left_edges, widths, save_dir)

        # plot hit metrics
        print("Plotting hit metrics ...")
        
        plot_hit_metrics(
            hit_metrics['eta_eff'], 
            hit_metrics['pt_eff'], 
            hit_metrics['eta_pur'], 
            hit_metrics['eta_bins'], 
            hit_metrics['pt_bins'], 
            save_dir,
            metrics
        )

        pl_module.metrics.reset()
