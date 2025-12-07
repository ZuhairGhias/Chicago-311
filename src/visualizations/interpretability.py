import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance_comparison(models_dict, feature_names, save_path='results/interpretability/feature_importance.png'):
    # compare feature importance across multiple models
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6*len(models_dict), 6))

    if len(models_dict) == 1:
        axes = [axes]

    for idx, (model_name, importance_df) in enumerate(models_dict.items()):
        ax = axes[idx]

        # sort by importance
        importance_df_sorted = importance_df.sort_values('importance', ascending=True).tail(15)

        ax.barh(range(len(importance_df_sorted)), importance_df_sorted['importance'],
                color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(importance_df_sorted)))
        ax.set_yticklabels(importance_df_sorted['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'{model_name}\nFeature Importance', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        # add values
        for i, v in enumerate(importance_df_sorted['importance']):
            ax.text(v + 0.01*max(importance_df_sorted['importance']), i,
                   f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved: {save_path}")


def plot_shap_analysis(model, X_test, feature_names, model_name, save_path='results/interpretability/'):
    # create shap analysis plots
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    # create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # handle different shap_values formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'{model_name} - SHAP Summary Plot', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     plot_type='bar', show=False)
    plt.title(f'{model_name} - SHAP Feature Importance', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP plots saved for {model_name}")

    return shap_values


def plot_shap_dependence(shap_values, X_test, feature_names, top_features=3,
                        model_name='Model', save_path='results/interpretability/'):
    # plot shap dependence for top features
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    # get top features by mean absolute shap value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_features:]

    fig, axes = plt.subplots(1, top_features, figsize=(6*top_features, 5))
    if top_features == 1:
        axes = [axes]

    for i, idx in enumerate(top_idx):
        feature_name = feature_names[idx]
        shap.dependence_plot(idx, shap_values, X_test,
                            feature_names=feature_names,
                            ax=axes[i], show=False)
        axes[i].set_title(f'SHAP Dependence: {feature_name}', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP dependence plots saved for {model_name}")
