import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_clustering_results():
    filepath = 'rfm_data_with_clusters.csv'
    
    if not os.path.exists(filepath):
        print(f"Khong tim thay file {filepath}. Vui long chay pipeline de tao du lieu truoc.")
        return
        
    print(f"Dang doc du lieu tu {filepath}...")
    df = pd.read_csv(filepath)
    
    # Kiem tra cac cot can thiet
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Cluster']
    for col in required_cols:
        if col not in df.columns:
            print(f"Loi: Thieu cot {col} trong du lieu!")
            return

    # ==========================================
    # Bieu do 1: Phan phoi khach hang theo cum
    # ==========================================
    plt.figure(figsize=(8, 6))
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors[:len(cluster_counts)])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(cluster_counts.values) * 0.01), 
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Số Lượng Khách Hàng Theo Từng Cụm (K-Means)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Cụm (Cluster)', fontsize=12)
    plt.ylabel('Số lượng khách hàng', fontsize=12)
    plt.tight_layout()
    plt.savefig('bieu_do_3_phan_phoi_cum.png', dpi=300)
    print("Da luu: bieu_do_3_phan_phoi_cum.png")

    # ==========================================
    # Bieu do 2: Boxplot so sanh R, F, M giua cac cum
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    features = ['Recency', 'Frequency', 'Monetary']
    titles = ['Recency (Số ngày từ lần cuối mua)', 'Frequency (Tần suất mua)', 'Monetary (Tổng tiền chi tiêu)']
    
    for i, feature in enumerate(features):
        data_by_cluster = [df[df['Cluster'] == c][feature].dropna().values for c in sorted(df['Cluster'].unique())]
        axes[i].boxplot(data_by_cluster, labels=sorted(df['Cluster'].unique()), patch_artist=True,
                        boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                        medianprops=dict(color='red', linewidth=2))
            
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Giá trị')
        # Gioi han truc y de loai bo bot outlier cho bieu do de nhin
        q3 = df[feature].quantile(0.95)
        if q3 > 0:
            axes[i].set_ylim(0, q3 * 2.5)

    plt.suptitle('Đặc Trưng Của Các Cụm Khách Hàng (Boxplot)', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('bieu_do_4_boxplot_cum.png', dpi=300, bbox_inches='tight')
    print("Da luu: bieu_do_4_boxplot_cum.png")

    # ==========================================
    # Bieu do 3: 3D Scatter Plot
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df['Recency'], df['Frequency'], df['Monetary'], 
                         c=df['Cluster'], cmap='viridis', alpha=0.6, edgecolors='w', s=30)
    
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Phân Bố Cụm Khách Hàng Trong Không Gian 3D (R-F-M)', fontsize=14, fontweight='bold', pad=20)
    
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    plt.savefig('bieu_do_5_3d_scatter.png', dpi=300)
    print("Da luu: bieu_do_5_3d_scatter.png")

    plt.show()

if __name__ == "__main__":
    plot_clustering_results()
