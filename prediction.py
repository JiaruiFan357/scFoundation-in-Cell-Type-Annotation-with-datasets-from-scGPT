# Core imports
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import scanpy as sc
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
sys.path.append("../model/")

from load import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import umap


def evaluate(model, data_loader, device):
    """
    Evaluate model performance on given data loader
    Returns accuracy, F1 score, and prediction details
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model({'x': data, 'targets': target})
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    # f1 = f1_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return accuracy, f1, all_targets, all_preds


def prepare_data(expression_csv, adata, label_encoder):
    df = pd.read_csv(expression_csv, index_col=0)
    X = df.values
    labels = adata.obs['Celltype']
    
    if label_encoder is None:
        raise ValueError("A pre-fitted LabelEncoder must be provided!")

    y = label_encoder.transform(labels)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return TensorDataset(X_tensor, y_tensor)


class LinearProbingClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes, frozenmore=True):
        
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.n_classes = n_classes
        
    def build(self):
        """Build model architecture and load weights"""
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        # Freeze encoder layers except second to last
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            param.requires_grad = True
            print('self.encoder.transformer_encoder', na, 'have grad')
            
        # Classification head
        hidden_dim = model_config['encoder']['hidden_dim']
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.n_classes)
        ) 
        
        self.norm = torch.nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, sample_list, *args, **kwargs):
        """Forward pass of the model"""
        label = sample_list['targets']
        x = sample_list['x']  # (B, L)
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                          self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb
        
        # Encoder and classification
        logits = self.encoder(x, x_padding)
        logits, _ = torch.max(logits, dim=1)  # b,dim
        logits = self.norm(logits)
        logits = self.fc1(logits)
        
        return logits


def main():
    train_adata = sc.read_h5ad('/hPancreas_train_seed_0.h5ad')
    val_adata = sc.read_h5ad('/hPancreas_val_seed_0.h5ad')
    test_adata =sc.read_h5ad('/hPancreas_test.h5ad')
     

    # Before using prepare_data, load or define global_label_encoder
    global_label_encoder = LabelEncoder()
    all_cell_types = pd.concat([
        train_adata.obs['Celltype'],
        val_adata.obs['Celltype'],
        test_adata.obs['Celltype']
    ])
    global_label_encoder.fit(all_cell_types)

    # Prepare datasets
    test_dataset = prepare_data('/hPancreas_test.csv', test_adata, global_label_encoder)
    train_dataset = prepare_data(f'/hPancreas_train_seed_0.csv', train_adata, global_label_encoder)
    val_dataset = prepare_data(f'/hPancreas_val_seed_0.csv', val_adata, global_label_encoder)
      
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)


    best_model_path = '/model_epoch_7.pth'
    n_classes = len(global_label_encoder.classes_)
    best_model = LinearProbingClassifier(
        ckpt_path='../models.ckpt',
        n_classes=n_classes
    )
    best_model.build()
    # Proper checkpoint loading (handle missing/unexpected keys)
    checkpoint = torch.load(best_model_path, weights_only=True)
    best_model.load_state_dict(checkpoint, strict=False)  
    best_model = best_model.cuda()

    test_accuracy, test_f1, all_targets, all_preds = evaluate(best_model, test_loader, torch.device("cuda"))
    print(f"Best Model - Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}")



    y_train = train_dataset.tensors[1].numpy()
    y_test = test_dataset.tensors[1].numpy()
    # Calculate distribution of labels in both datasets
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)

    # Get mapping from numeric labels to cell types
    label_to_celltype = dict(enumerate(train_adata.obs['Celltype'].cat.categories))

    # Create a wider figure to accommodate the legend
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(15, 6), 
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # Plot bar charts in the left subplot
    ax1.bar(
        train_unique - 0.2, 
        train_counts, 
        width=0.4, 
        label='Train_data', 
        color='skyblue', 
        align='center'
    )
    ax1.bar(
        test_unique + 0.2, 
        test_counts, 
        width=0.4, 
        label='Test_data', 
        color='salmon', 
        align='center'
    )

    # Set labels and title for the left subplot
    ax1.set_xlabel('Unique Labels')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Train_data and Test_data')
    ax1.set_xticks(np.union1d(train_unique, test_unique))
    ax1.legend()

    # Create legend in the right subplot
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'{k}: {v}',
            markerfacecolor='gray',
            markersize=10
        ) 
        for k, v in label_to_celltype.items()
    ]

    # Configure right subplot for legend display
    ax2.legend(
        handles=legend_elements,
        title='Label: Celltype',
        loc='center'
    )
    ax2.axis('off')  # Hide axes

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('Distribution1.png')
    plt.show()




    # Calculate unique values and their counts for y_train, y_test, and pre
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    pre_unique, pre_counts = np.unique(all_preds, return_counts=True)

    # Plot the distribution for y_train, y_test, and pre
    plt.figure(figsize=(12, 6))
    bar_width = 0.25

    plt.bar(train_unique - bar_width, train_counts, width=bar_width, label='Train_data', color='skyblue', align='center')
    plt.bar(test_unique, test_counts, width=bar_width, label='Test_data', color='salmon', align='center')
    plt.bar(pre_unique + bar_width, pre_counts, width=bar_width, label='Prediction', color='lightgreen', align='center')

    # Adding labels and title
    plt.xlabel('Unique Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Train_data, Test_data, and Prediction')
    plt.xticks(np.unique(np.concatenate((train_unique, test_unique, pre_unique))))
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Distribution2.png')
    plt.show()



    # Get mapping from labels to cell types
    label_to_celltype = dict(enumerate(train_adata.obs['Celltype'].cat.categories))

    # Create figure with wider layout to accommodate legend
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(20, 8), 
        gridspec_kw={'width_ratios': [4, 1]}
    )

    # Plot grouped bar charts
    bar_width = 0.25
    ax1.bar(
        train_unique - bar_width,
        train_counts,
        width=bar_width,
        label='Train_data',
        color='skyblue',
        align='center'
    )
    ax1.bar(
        test_unique,
        test_counts,
        width=bar_width,
        label='Test_data',
        color='salmon',
        align='center'
    )
    ax1.bar(
        pre_unique + bar_width,
        pre_counts,
        width=bar_width,
        label='Prediction',
        color='lightgreen',
        align='center'
    )

    # Configure main plot labels and title
    ax1.set_xlabel('Unique Labels')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Train_data, Test_data, and Prediction')
    ax1.set_xticks(np.unique(np.concatenate((train_unique, test_unique, pre_unique))))
    ax1.legend()

    # Create legend mapping in the right subplot
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'{k}: {v}',
            markerfacecolor='gray',
            markersize=10
        ) 
        for k, v in label_to_celltype.items()
    ]

    # Configure legend display
    ax2.legend(
        handles=legend_elements,
        title='Label: Celltype',
        loc='center',
        fontsize='small'
    )
    ax2.axis('off')  # Hide axes

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Distribution.png')
    plt.show()




    def compare_distributions(dist1, dist2, dist3, names):
        distributions = [dist1, dist2, dist3]
        n = len(names)
        
        # Initialize results storage
        results = {
            'KS Test': np.zeros((n, n)),
            'Jensen-Shannon': np.zeros((n, n)),
            'Wasserstein': np.zeros((n, n))
        }
        
        for i in range(n):
            for j in range(i+1, n):
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(distributions[i], distributions[j])
                results['KS Test'][i, j] = ks_statistic
                results['KS Test'][j, i] = p_value
                
                # Calculate probability distributions
                hist1, _ = np.histogram(distributions[i], bins=50, density=True)
                hist2, _ = np.histogram(distributions[j], bins=50, density=True)
                
                # Jensen-Shannon divergence
                js_div = jensenshannon(hist1, hist2)
                results['Jensen-Shannon'][i, j] = results['Jensen-Shannon'][j, i] = js_div
                
                # Wasserstein distance
                w_dist = wasserstein_distance(distributions[i], distributions[j])
                results['Wasserstein'][i, j] = results['Wasserstein'][j, i] = w_dist
        
        return results

    names = ['Train_data', 'Test_data', 'Prediction']
    results = compare_distributions(y_train, y_test, all_preds, names)
    
    # Write results to file and print them
    with open("comparison_results.txt", "w") as f:
        for method, matrix in results.items():
            f.write(f"{method}:\n")
            df_matrix = pd.DataFrame(matrix, index=names, columns=names)
            f.write(df_matrix.to_string())
            f.write("\n\n")
            print(f"{method}:")
            print(df_matrix)
            print("\n")


    print("test_adata shape:", test_adata.shape[0])
    print("Number of predictions:", len(all_preds)) 

    # Generate confusion matrix explicitly
    cm = confusion_matrix(all_targets, all_preds, normalize='true')

    # Get unique classes explicitly
    unique_labels = np.unique(np.concatenate([all_targets, all_preds]))

    # Extract the correct class labels explicitly
    label_names = global_label_encoder.inverse_transform(unique_labels)

    # Plot explicitly and robustly:
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(ax=ax, xticks_rotation='vertical')

    # Explicitly set tick labels to cell-type names
    ax.set_xticklabels([global_label_encoder.classes_[i] for i in unique_labels], rotation=90)
    ax.set_yticklabels([global_label_encoder.classes_[i] for i in unique_labels])

    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


    # UMAP visualization of predictions
    X_test = test_dataset.tensors[0].numpy()

    # Fit UMAP on test data
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.1, n_components=2, random_state=42)
    X_test_umap = reducer.fit_transform(X_test)

    # Plot UMAP embeddings with predicted labels
    # Updated and consistent UMAP visualization
    # test_adata.obs['batch'] = 'Test'

    palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3
    celltype_labels = global_label_encoder.classes_
    palette_ = {c: palette_colors[i] for i, c in enumerate(celltype_labels)}

    # UMAP visualization
    sc.pp.neighbors(test_adata, use_rep='X')
    sc.tl.umap(test_adata)

    # Add predictions
    test_adata.obs["predictions"] = global_label_encoder.inverse_transform(all_preds)

    # Visualize original and predicted labels
    with plt.rc_context({"figure.figsize": (12, 10), "figure.dpi": 300}):
        sc.pl.umap(
            test_adata,
            color=["Celltype", "predictions"],
            palette=palette_,
            wspace=0.3,
            show=False,
        )
        plt.savefig("UMAP_results.png", dpi=300)
        plt.show()

    # with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": 300}):
    #     sc.pl.umap(
    #         test_adata,
    #         color=["Celltype", "predictions", "batch"],
    #         palette=palette_,
    #         wspace=0.3,
    #         frameon=False,
    #         title=["Cell type", "Predicted Cell type", "Batch Label"],
    #         show=False
    #     )
    #     plt.savefig("UMAP_results2.png", dpi=300)
    #     plt.show()

if __name__ == '__main__':
    main()

