import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data():
    print("Loading data...")
    # Load data
    try:
        # feature-table.tsv has a comment line at the top
        feature_table = pd.read_csv('feature-table.tsv', sep='\t', skiprows=1, index_col=0)
        metadata = pd.read_csv('metadata.tsv', sep='\t')
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    # Transpose feature table: rows=samples, cols=genera
    X = feature_table.T
    
    # Ensure SampleID is the index for metadata to standardise merging
    if '#SampleID' in metadata.columns:
        metadata.set_index('#SampleID', inplace=True)
    
    # Merge
    # Keep only samples present in both
    common_samples = X.index.intersection(metadata.index)
    X = X.loc[common_samples]
    y = metadata.loc[common_samples, 'Condition']
    
    print(f"Dataset shape after merging: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Handle missing values in features (fill with 0 for abundance)
    X = X.fillna(0)
    
    # Normalize features (Relative Abundance)
    # Divide each row by its sum
    row_sums = X.sum(axis=1)
    X = X.div(row_sums, axis=0)
    
    # Sanitize feature names for XGBoost
    import re
    X.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X.columns]
    
    return X, y

def train_and_evaluate(X, y, task_name, class_labels):
    print(f"\n--- Starting {task_name} Classification ---")
    
    # Encoders
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=RANDOM_STATE)
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Determine prediction method for ROC
        if task_name == "Multiclass":
             # sklearn needs 0, 1, 2 for multiclass standard
             pass
        
        # We need true labels and predicted probabilities for ROC
        # Using cross_val_predict for simplicity in generating aggregate reports, 
        # but for correct ROC curves per fold, we usually iterate.
        # However, to produce a single 'robust' result as requested, iterating is better.
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        y_true_all = []
        y_pred_all = []
        y_proba_all = []
        
        importances = []
        
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)
            
            # Feature Importance
            if model_name == 'RandomForest':
                importances.append(model.feature_importances_)
            elif model_name == 'XGBoost':
                importances.append(model.feature_importances_)
        
        # Aggregate Metrics
        acc = accuracy_score(y_true_all, y_pred_all)
        
        if task_name == 'Binary':
            # binary roc_auc needs proba of positive class (usually index 1)
            # Assuming 'OSCC' is the positive class or simply the second one.
            # Let's check classes order
            y_proba_all_np = np.array(y_proba_all) # shape (N, 2)
            roc_auc = roc_auc_score(y_true_all, y_proba_all_np[:, 1])
        else:
            # Multiclass
            roc_auc = roc_auc_score(y_true_all, y_proba_all, multi_class='ovr', average='macro')
            
        report = classification_report(y_true_all, y_pred_all, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(y_true_all, y_pred_all)
        
        results[model_name] = {
            'Accuracy': acc,
            'ROC_AUC': roc_auc,
            'Report': report,
            'Confusion_Matrix': cm,
            'Feature_Importances': np.mean(importances, axis=0) if importances else None,
            'y_true': y_true_all,
            'y_proba': np.array(y_proba_all),
            'Classes': le.classes_
        }
        
        print(f"  {model_name}: Accuracy={acc:.4f}, AUC={roc_auc:.4f}")
        
    return results

def plot_results(results, X, task_name):
    timestamp = task_name.lower().replace(" ", "_")
    
    for model_name, res in results.items():
        classes = res['Classes']
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(res['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix: {task_name} - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/cm_{timestamp}_{model_name}.png')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        y_true = np.array(res['y_true'])
        y_proba = res['y_proba']
        
        if task_name == 'Binary':
            # Binary
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
        else:
            # Multiclass (One-vs-Rest)
            for i, class_label in enumerate(classes):
                # Create binary labels for current class
                y_true_bin = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {task_name} - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/roc_{timestamp}_{model_name}.png')
        plt.close()
        
        # 3. Feature Importance
        if res['Feature_Importances'] is not None:
            feat_imp = pd.Series(res['Feature_Importances'], index=X.columns)
            top_20 = feat_imp.nlargest(20)
            
            plt.figure(figsize=(10, 8))
            top_20.sort_values().plot(kind='barh')
            plt.title(f'Top 20 Feature Importances: {task_name} - {model_name}')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/features_{timestamp}_{model_name}.png')
            plt.close()

    # Save summary metrics to text file
    with open(f'{RESULTS_DIR}/metrics_summary.txt', 'a') as f:
        f.write(f"\n\n=== {task_name} Classification Results ===\n")
        for model_name, res in results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write(f"Accuracy: {res['Accuracy']:.4f}\n")
            f.write(f"ROC AUC: {res['ROC_AUC']:.4f}\n")
            f.write("Classification Report:\n")
            # Reformat report dict to string
            report_df = pd.DataFrame(res['Report']).transpose()
            f.write(report_df.to_string())
            f.write("\n")

def main():
    X, y = load_and_preprocess_data()
    if X is None:
        return

    # Clear previous metrics file
    if os.path.exists(f'{RESULTS_DIR}/metrics_summary.txt'):
        os.remove(f'{RESULTS_DIR}/metrics_summary.txt')

    # --- Scenario 1: Multiclass ---
    print("\nProcessing Multiclass (Healthy, PreCancer, OSCC)...")
    res_multi = train_and_evaluate(X, y, "Multiclass", class_labels=None)
    plot_results(res_multi, X, "Multiclass")

    # --- Scenario 2: Binary (Healthy vs OSCC) ---
    print("\nProcessing Binary (Healthy vs OSCC)...")
    # Filter data
    mask = y.isin(['Healthy', 'OSCC'])
    X_bin = X[mask]
    y_bin = y[mask]
    
    print(f"Binary Dataset shape: {X_bin.shape}")
    print(f"Binary Class distribution:\n{y_bin.value_counts()}")

    res_binary = train_and_evaluate(X_bin, y_bin, "Binary", class_labels=['Healthy', 'OSCC'])
    plot_results(res_binary, X_bin, "Binary")

    print(f"\nDone! Results saved to available in '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()
