import numpy as np
import json


def load_jsonl(file_path):
    """加载 JSONL 文件。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]


def save_jsonl(data, file_path):
    """保存数据到 JSONL 文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def analyze_grid_distribution(points, grid_size=100):
    """
    将数据点分布到二维网格中，并统计每个小格子的点数量。
    参数：
    - grid_size: 网格的大小 (grid_size x grid_size)
    返回：
    - grid_distribution: 字典，记录每种点数量对应的网格数目。
    - counts: 2D 数组，每个小格子的点数量。
    - grid_info: 字典，记录每个小格子包含的点索引。
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    # 获取坐标范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    # 使用 np.histogram2d 统计每个网格的点数量
    counts, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=grid_size,
                                              range=[[x_min, x_max], [y_min, y_max]])
    counts = counts.astype(int)
    grid_info = {}
    for idx, (x, y) in enumerate(points):
        row = np.searchsorted(x_edges, x, side='right') - 1
        col = np.searchsorted(y_edges, y, side='right') - 1
        if (row, col) not in grid_info:
            grid_info[(row, col)] = []
        grid_info[(row, col)].append(idx)
    return counts, grid_info


def select_n_samples_with_priority(data, grid_info, target_total):
    """
    按优先级规则从二维网格中逐步选取数据。
    参数：
    - data: 输入数据列表，每个元素是字典，包含 'x_tsne', 'y_tsne' 和 'original_index'。
    - grid_info: 小格子分布信息，格式为 {(row, col): [index1, index2, ...]}。
    - target_total: 目标提取的总数量。
    返回：
    - selected_data: 选中的数据列表。
    """
    # 对网格按照点数量从大到小排序
    sorted_cells = sorted(grid_info.items(), key=lambda x: len(x[1]), reverse=True)

    selected_indices = set()
    total_selected = 0

    for rank, (cell, indices) in enumerate(sorted_cells):
        if total_selected >= target_total:
            break
        # 确定当前格子的取点数量规则
        if rank < 1000:  # Top 1000
            to_select = min(5, len(indices))
        elif rank < 2000:  # Top 1000-2000
            to_select = min(3, len(indices))
        else:  # 其他
            to_select = min(1, len(indices))

        # 选择数据
        selected_indices.update(indices[:to_select])
        total_selected = len(selected_indices)

        # 如果达到目标总数，提前退出
        if total_selected >= target_total:
            break

    selected_data = [data[i] for i in selected_indices]
    return selected_data


# 主函数
if __name__ == "__main__":
    grid_size = 50
    target_total = 5000
    input_file = '/share/project/chengweiwu/code/code_synthesis/Scaling_regularity_guided_instruction_synthesize/data/AAAI_set/AAAI_set_900w_pipei_304w_with_coords_sort.jsonl'
    output_file = '/share/project/chengweiwu/code/code_synthesis/Scaling_regularity_guided_instruction_synthesize/code/scaling_low_pick_data/small_data_pick_data/shiyan3/AAAI_set_900w_pipei_304w_with_coords_sort_scalinglaw_5k_30X30.jsonl'

    # 加载数据
    data = load_jsonl(input_file)

    # 提取点的二维坐标
    points = np.array([[item['x_tsne'], item['y_tsne']] for item in data])

    # 分析网格分布
    counts, grid_info = analyze_grid_distribution(points, grid_size)

    # 按新规则多轮挑选数据
    selected_data = select_n_samples_with_priority(data, grid_info, target_total)

    # 保存到新的 jsonl 文件
    save_jsonl(selected_data, output_file)

    print(f"Successfully saved {len(selected_data)} samples to {output_file}")
