import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_hierarchical_results():
    if not os.path.exists('final_data.csv') or not os.path.exists('scaled_data_hierarchical.csv'):
        print("Loi: Thieu file du lieu. Vui long chay pipeline truoc.")
        return
        
    print("Dang doc du lieu...")
    df_orig = pd.read_csv('final_data.csv')
    df_labels = pd.read_csv('scaled_data_hierarchical.csv')[['CustomerID', 'Cluster_Label']]
    
    df = df_orig.merge(df_labels, on='CustomerID', how='inner')
    
    # Rename for consistency
    df.rename(columns={'Cluster_Label': 'Cluster'}, inplace=True)
    
    # Bieu do 1: Phan phoi khach hang
    plt.figure(figsize=(8, 6))
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors[:len(cluster_counts)])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(cluster_counts.values) * 0.01), 
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Số Lượng Khách Hàng Theo Từng Cụm (Hierarchical)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Cụm (Cluster)', fontsize=12)
    plt.ylabel('Số lượng khách hàng', fontsize=12)
    plt.tight_layout()
    plt.savefig('bieu_do_6_phan_phoi_cum_h.png', dpi=300)
    print("Da luu: bieu_do_6_phan_phoi_cum_h.png")

    # Bieu do 2: Boxplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    features = ['Recency', 'Frequency', 'Monetary']
    titles = ['Recency (Số ngày từ lần cuối mua)', 'Frequency (Tần suất mua)', 'Monetary (Tổng tiền chi tiêu)']
    
    for i, feature in enumerate(features):
        data_by_cluster = [df[df['Cluster'] == c][feature].dropna().values for c in sorted(df['Cluster'].unique())]
        axes[i].boxplot(data_by_cluster, tick_labels=sorted(df['Cluster'].unique()), patch_artist=True,
                        boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                        medianprops=dict(color='red', linewidth=2))
            
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Giá trị')
        
        q3 = df[feature].quantile(0.95)
        if q3 > 0:
            axes[i].set_ylim(0, q3 * 2.5)

    plt.suptitle('Đặc Trưng Của Các Cụm Khách Hàng - Hierarchical (Boxplot)', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('bieu_do_7_boxplot_cum_h.png', dpi=300, bbox_inches='tight')
    print("Da luu: bieu_do_7_boxplot_cum_h.png")

    # Bieu do 3: 3D Scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Chuyen chuoi I, II, III thanh so cho cmap de dang
    cluster_mapping = {c: i for i, c in enumerate(sorted(df['Cluster'].unique()))}
    colors_numeric = df['Cluster'].map(cluster_mapping)
    
    scatter = ax.scatter(df['Recency'], df['Frequency'], df['Monetary'], 
                         c=colors_numeric, cmap='viridis', alpha=0.6, edgecolors='w', s=30)
    
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Phân Bố Cụm Khách Hàng (Hierarchical) Trong Không Gian 3D', fontsize=14, fontweight='bold', pad=20)
    
    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, cluster_mapping.keys(), title="Clusters")
    ax.add_artist(legend)
    
    plt.savefig('bieu_do_8_3d_scatter_h.png', dpi=300)
    print("Da luu: bieu_do_8_3d_scatter_h.png")

    plt.show()

if __name__ == "__main__":
    plot_hierarchical_results()
