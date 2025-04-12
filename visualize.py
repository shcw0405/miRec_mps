import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无GUI环境中出错
import random

def visualize_item_embeddings_tsne(model, dataset, model_type, hidden_size, args, sample_size=2000, perplexity=50, n_iter=1000, cate_file=None, external_embeddings=None, force_no_sampling=False, point_size=15, figure_size=(16, 14), early_exaggeration=12.0):
    """
    使用t-SNE可视化item embeddings
    
    参数:
    - model: 训练好的模型
    - dataset: 数据集名称
    - model_type: 模型类型
    - hidden_size: 隐藏层维度
    - args: 参数
    - sample_size: 采样大小，如果为None或force_no_sampling=True则使用所有item
    - perplexity: t-SNE的perplexity参数
    - n_iter: t-SNE的迭代次数
    - cate_file: 物品类别文件路径，如果提供则按类别着色
    - external_embeddings: 外部embedding列表，将与模型的item embedding一起可视化，用红色标记
    - force_no_sampling: 是否强制不进行采样，为True时将可视化所有item embedding
    """
    print("开始可视化item embeddings...")
    
    # 创建可视化目录
    vis_dir = os.path.join("visualization", f"{dataset}_{model_type}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取item embeddings
    if hasattr(model, 'item_embeddings'):
        item_embeddings = model.item_embeddings.weight.cpu().detach().numpy()
        print(f"Item embeddings shape: {item_embeddings.shape}")
        
        # 排除padding item (index 0)
        item_embeddings = item_embeddings[1:]
        item_indices = np.arange(1, item_embeddings.shape[0] + 1)
        
        # 采样，如果item数量太多且不强制不采样
        if not force_no_sampling and sample_size is not None and item_embeddings.shape[0] > sample_size:
            print(f"采样 {sample_size} 个items进行可视化")
            sample_indices = random.sample(range(item_embeddings.shape[0]), sample_size)
            item_embeddings = item_embeddings[sample_indices]
            item_indices = item_indices[sample_indices]
        elif force_no_sampling:
            print(f"强制不采样，可视化所有 {item_embeddings.shape[0]} 个items")
        
        # 处理外部embeddings
        has_external_embeddings = external_embeddings is not None and len(external_embeddings) > 0
        combined_embeddings = item_embeddings
        external_indices = []
        
        if has_external_embeddings:
            print(f"添加 {len(external_embeddings)} 个外部embeddings进行可视化")
            # 确保外部embeddings是numpy数组且维度匹配
            external_embeddings_np = np.array(external_embeddings)
            if external_embeddings_np.ndim == 1:
                # 如果是单个embedding，转换为2D数组
                external_embeddings_np = external_embeddings_np.reshape(1, -1)
            
            # 记录外部embeddings的索引位置
            external_indices = list(range(len(item_embeddings), len(item_embeddings) + len(external_embeddings_np)))
            
            # 合并item embeddings和外部embeddings
            combined_embeddings = np.vstack([item_embeddings, external_embeddings_np])
        
        # 使用t-SNE降维
        print(f"使用t-SNE进行降维 (perplexity={perplexity}, n_iter={n_iter}, early_exaggeration={early_exaggeration})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, early_exaggeration=early_exaggeration)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        # 分离item embeddings和外部embeddings的降维结果
        item_embeddings_2d = embeddings_2d[:len(item_embeddings)]
        external_embeddings_2d = embeddings_2d[len(item_embeddings):] if has_external_embeddings else None
        
        # 绘制散点图
        plt.figure(figsize=figure_size)
        
        # 如果有外部embeddings，先绘制它们（红色）
        if has_external_embeddings and external_embeddings_2d is not None and len(external_embeddings_2d) > 0:
            plt.scatter(
                external_embeddings_2d[:, 0],
                external_embeddings_2d[:, 1],
                c='red',
                label='External Embeddings',
                alpha=0.9,
                s=point_size*2,  # 外部嵌入点稍大一些
                marker='*'
            )
        
        # 如果提供了类别文件，按类别着色
        if cate_file is not None:
            from utils import load_item_cate
            item_cate_map = load_item_cate(cate_file)
            
            # 获取可视化items的类别
            categories = []
            valid_indices = []
            for i, idx in enumerate(item_indices):
                if idx in item_cate_map:
                    categories.append(item_cate_map[idx])
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                # 只保留有类别的items
                item_embeddings_2d = item_embeddings_2d[valid_indices]
                item_indices = item_indices[valid_indices]
                
                # 获取唯一类别并分配颜色
                unique_categories = list(set(categories))
                print(f"发现 {len(unique_categories)} 个不同的物品类别")
                
                # 限制类别数量，如果太多
                max_categories_to_show = 20
                if len(unique_categories) > max_categories_to_show:
                    # 找出最常见的类别
                    from collections import Counter
                    category_counts = Counter(categories)
                    top_categories = [cat for cat, _ in category_counts.most_common(max_categories_to_show)]
                    other_category = -1  # 用于表示"其他"类别
                    
                    # 重新映射类别
                    for i in range(len(categories)):
                        if categories[i] not in top_categories:
                            categories[i] = other_category
                    
                    unique_categories = top_categories + [other_category]
                    print(f"限制为显示前 {max_categories_to_show} 个最常见的类别")
                
                # 创建类别到颜色的映射
                cmap = plt.cm.get_cmap('tab20', len(unique_categories))
                category_to_color = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
                
                # 按类别绘制散点图
                for cat in unique_categories:
                    if cat == -1:  # "其他"类别
                        label = "其他"
                    else:
                        label = f"cate {cat}"
                    
                    indices = [i for i, c in enumerate(categories) if c == cat]
                    plt.scatter(
                        item_embeddings_2d[indices, 0],
                        item_embeddings_2d[indices, 1],
                        c=[category_to_color[cat]],
                        label=label,
                        alpha=0.7,
                        s=point_size
                    )
                
                # 如果有外部embeddings，确保图例包含它们
                if has_external_embeddings:
                    # 获取当前图例句柄和标签
                    handles, labels = plt.gca().get_legend_handles_labels()
                    # 确保外部embeddings的图例在最前面
                    if 'External Embeddings' not in labels:
                        red_patch = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='External Embeddings')
                        handles = [red_patch] + handles
                        labels = ['External Embeddings'] + labels
                    plt.legend(handles=handles, labels=labels, loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
                else:
                    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
            else:
                print("没有找到物品类别信息，使用默认颜色")
                plt.scatter(item_embeddings_2d[:, 0], item_embeddings_2d[:, 1], alpha=0.7, s=point_size)
        else:
            # 不按类别着色
            plt.scatter(item_embeddings_2d[:, 0], item_embeddings_2d[:, 1], alpha=0.7, s=point_size, label='Item Embeddings')
            
            # 如果有外部embeddings，添加图例
            if has_external_embeddings:
                plt.legend(loc='best')
        
        plt.title(f'Item Embeddings Visualization ({model_type} on {dataset})')
        plt.xlabel('t-SNE d 1')
        plt.ylabel('t-SNE d 2')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(vis_dir, f'item_embeddings_tsne.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE可视化已保存至: {save_path}")
        
        # 保存降维后的数据，以便后续分析
        save_data = {
            'embeddings_2d': item_embeddings_2d,
            'item_indices': item_indices,
            'categories': categories if cate_file is not None and len(valid_indices) > 0 else None
        }
        
        # 如果有外部embeddings，也保存它们
        if has_external_embeddings:
            save_data['external_embeddings_2d'] = external_embeddings_2d
        
        np.save(os.path.join(vis_dir, 'item_embeddings_tsne.npy'), save_data)
    else:
        print("模型没有item_embeddings属性，无法可视化")