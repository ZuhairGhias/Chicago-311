import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_correlation_heatmap(df, features, save_path='results/eda/correlation_heatmap.png'):
    # create correlation heatmap for numerical and encoded categorical features
    fig, ax = plt.subplots(figsize=(10, 8))

    # select only numeric columns for correlation
    numeric_df = df[features].select_dtypes(include=[np.number])

    correlation_matrix = numeric_df.corr()

    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return correlation_matrix


def plot_temporal_trends(df, time_col='CREATED_DATE', target_col='RESPONSE_TIME_DAYS',
                        save_path='results/eda/temporal_trends.png'):
    # plot response time trends over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy = df_copy.sort_values(time_col)

    # monthly average
    df_copy['year_month'] = df_copy[time_col].dt.to_period('M')
    monthly_avg = df_copy.groupby('year_month')[target_col].agg(['mean', 'median', 'std']).reset_index()
    monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()

    ax1.plot(monthly_avg['year_month'], monthly_avg['mean'], marker='o', label='Mean', linewidth=2)
    ax1.plot(monthly_avg['year_month'], monthly_avg['median'], marker='s', label='Median', linewidth=2)
    ax1.fill_between(monthly_avg['year_month'],
                      monthly_avg['mean'] - monthly_avg['std'],
                      monthly_avg['mean'] + monthly_avg['std'],
                      alpha=0.3, label='±1 Std Dev')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Response Time (days)', fontweight='bold')
    ax1.set_title('Monthly Response Time Trends', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # by day of week
    dow_avg = df_copy.groupby('CREATED_DAY_OF_WEEK')[target_col].mean().reset_index()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax2.bar(dow_names, dow_avg[target_col], color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Day of Week', fontweight='bold')
    ax2.set_ylabel('Avg Response Time (days)', fontweight='bold')
    ax2.set_title('Average Response Time by Day of Week', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # by hour of day
    hour_avg = df_copy.groupby('CREATED_HOUR')[target_col].mean().reset_index()
    ax3.plot(hour_avg['CREATED_HOUR'], hour_avg[target_col], marker='o', linewidth=2, color='darkgreen')
    ax3.set_xlabel('Hour of Day', fontweight='bold')
    ax3.set_ylabel('Avg Response Time (days)', fontweight='bold')
    ax3.set_title('Average Response Time by Hour of Day', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_distribution(df, lat_col='LATITUDE', lon_col='LONGITUDE',
                              target_col='RESPONSE_TIME_DAYS',
                              save_path='results/eda/spatial_distribution.png'):
    # plot spatial distribution of response times
    df_clean = df[[lat_col, lon_col, target_col]].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # scatter plot colored by response time
    scatter = ax1.scatter(df_clean[lon_col], df_clean[lat_col],
                         c=df_clean[target_col], cmap='YlOrRd',
                         alpha=0.6, s=10, edgecolors='none')
    ax1.set_xlabel('Longitude', fontweight='bold')
    ax1.set_ylabel('Latitude', fontweight='bold')
    ax1.set_title('Geographic Distribution of Response Times', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Response Time (days)', fontweight='bold')
    ax1.grid(alpha=0.3)

    # hexbin density plot
    hexbin = ax2.hexbin(df_clean[lon_col], df_clean[lat_col],
                        C=df_clean[target_col], gridsize=30, cmap='YlOrRd',
                        reduce_C_function=np.mean)
    ax2.set_xlabel('Longitude', fontweight='bold')
    ax2.set_ylabel('Latitude', fontweight='bold')
    ax2.set_title('Average Response Time by Location (Hexbin)', fontweight='bold', fontsize=12)
    cbar2 = plt.colorbar(hexbin, ax=ax2)
    cbar2.set_label('Avg Response Time (days)', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_target_distribution(df, target_col='RESPONSE_TIME_DAYS',
                             save_path='results/eda/target_distribution.png'):
    # plot distribution of target variable
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # histogram
    ax1.hist(df[target_col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Response Time (days)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Response Times', fontweight='bold', fontsize=12)
    ax1.axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[target_col].mean():.1f}')
    ax1.axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[target_col].median():.1f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # log scale
    ax2.hist(np.log1p(df[target_col]), bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log(Response Time + 1)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Distribution of Log-Transformed Response Times', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # print statistics
    print(f"\nTarget Variable Statistics:")
    print(f"  Mean: {df[target_col].mean():.2f} days")
    print(f"  Median: {df[target_col].median():.2f} days")
    print(f"  Std Dev: {df[target_col].std():.2f} days")
    print(f"  Skewness: {df[target_col].skew():.2f}")
    print(f"  Min: {df[target_col].min():.2f} days")
    print(f"  Max: {df[target_col].max():.2f} days")
