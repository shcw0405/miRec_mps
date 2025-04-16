import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无GUI环境中出错
import random
from utils import get_exp_name

def visualize_item_embeddings_tsne(dataset, model_type, hidden_size, args, sample_size=2000, perplexity=50, n_iter=1000, cate_file=None, external_embeddings=None, force_no_sampling=False, point_size=15, figure_size=(16, 14), early_exaggeration=12.0):
    """
    使用t-SNE可视化item embeddings（直接从npy文件读取）
    """
    print("开始可视化item embeddings（从npy文件读取）...")
    # 创建可视化目录
    vis_dir = os.path.join("visualization", f"{dataset}_{model_type}")
    os.makedirs(vis_dir, exist_ok=True)

    # 使用utils.get_exp_name获取实验名和embedding路径
    exp_name = get_exp_name(args, save=False)
    emb_dir = os.path.join("best_model", exp_name)
    item_emb_path = os.path.join(emb_dir, "item_embeddings.npy")
    if not os.path.exists(item_emb_path):
        print(f"未找到item_embeddings.npy: {item_emb_path}")
        return
    item_embeddings = np.load(item_emb_path)
    print(f"Item embeddings shape: {item_embeddings.shape}")
    # 排除padding item (index 0)
    item_embeddings = item_embeddings[1:]
    item_indices = np.arange(1, item_embeddings.shape[0] + 1)
    # 采样
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
        external_embeddings_np = np.array(external_embeddings)
        if external_embeddings_np.ndim == 1:
            external_embeddings_np = external_embeddings_np.reshape(1, -1)
        external_indices = list(range(len(item_embeddings), len(item_embeddings) + len(external_embeddings_np)))
        combined_embeddings = np.vstack([item_embeddings, external_embeddings_np])
    # t-SNE降维
    print(f"使用t-SNE进行降维 (perplexity={perplexity}, n_iter={n_iter}, early_exaggeration={early_exaggeration})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, early_exaggeration=early_exaggeration)
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    item_embeddings_2d = embeddings_2d[:len(item_embeddings)]
    external_embeddings_2d = embeddings_2d[len(item_embeddings):] if has_external_embeddings else None
    plt.figure(figsize=figure_size)
    if has_external_embeddings and external_embeddings_2d is not None and len(external_embeddings_2d) > 0:
        plt.scatter(
            external_embeddings_2d[:, 0],
            external_embeddings_2d[:, 1],
            c='red',
            label='External Embeddings',
            alpha=0.9,
            s=point_size*2,
            marker='*'
        )
    categories = []
    valid_indices = []
    if cate_file is not None:
        from utils import load_item_cate
        item_cate_map = load_item_cate(cate_file)
        for i, idx in enumerate(item_indices):
            if idx in item_cate_map:
                categories.append(item_cate_map[idx])
                valid_indices.append(i)
        if len(valid_indices) > 0:
            item_embeddings_2d = item_embeddings_2d[valid_indices]
            item_indices = item_indices[valid_indices]
            unique_categories = list(set(categories))
            print(f"发现 {len(unique_categories)} 个不同的物品类别")
            max_categories_to_show = 20
            if len(unique_categories) > max_categories_to_show:
                from collections import Counter
                category_counts = Counter(categories)
                top_categories = [cat for cat, _ in category_counts.most_common(max_categories_to_show)]
                other_category = -1
                for i in range(len(categories)):
                    if categories[i] not in top_categories:
                        categories[i] = other_category
                unique_categories = top_categories + [other_category]
                print(f"限制为显示前 {max_categories_to_show} 个最常见的类别")
            cmap = plt.cm.get_cmap('tab20', len(unique_categories))
            category_to_color = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
            for cat in unique_categories:
                label = "其他" if cat == -1 else f"cate {cat}"
                indices = [i for i, c in enumerate(categories) if c == cat]
                plt.scatter(
                    item_embeddings_2d[indices, 0],
                    item_embeddings_2d[indices, 1],
                    c=[category_to_color[cat]],
                    label=label,
                    alpha=0.7,
                    s=point_size
                )
            if has_external_embeddings:
                handles, labels = plt.gca().get_legend_handles_labels()
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
        plt.scatter(item_embeddings_2d[:, 0], item_embeddings_2d[:, 1], alpha=0.7, s=point_size, label='Item Embeddings')
        if has_external_embeddings:
            plt.legend(loc='best')
    plt.title(f'Item Embeddings Visualization ({model_type} on {dataset})')
    plt.xlabel('t-SNE d 1')
    plt.ylabel('t-SNE d 2')
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'item_embeddings_tsne.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE可视化已保存至: {save_path}")
    save_data = {
        'embeddings_2d': item_embeddings_2d,
        'item_indices': item_indices,
        'categories': categories if cate_file is not None and len(valid_indices) > 0 else None
    }
    if has_external_embeddings:
        save_data['external_embeddings_2d'] = external_embeddings_2d
    np.save(os.path.join(vis_dir, 'item_embeddings_tsne.npy'), save_data)

def main():
    import argparse
    from utils import get_parser
    parser = get_parser()
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--perplexity', type=float, default=50)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--force_no_sampling', action='store_true')
    parser.add_argument('--point_size', type=int, default=15)
    parser.add_argument('--figure_size', type=int, nargs=2, default=[16, 14])
    parser.add_argument('--early_exaggeration', type=float, default=12.0)
    args = parser.parse_args()
    path = f'./data/{args.dataset}_data/'
    cate_file = path + args.dataset + '_item_cate.txt'
    visualize_item_embeddings_tsne(
        dataset=args.dataset,
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        args=args,
        sample_size=args.sample_size,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        cate_file=cate_file,
        external_embeddings=None,
        force_no_sampling=args.force_no_sampling,
        point_size=args.point_size,
        figure_size=tuple(args.figure_size),
        early_exaggeration=args.early_exaggeration
    )

if __name__ == '__main__':
    main()