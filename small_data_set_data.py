import numpy as np
import json


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    """将数据写入 jsonl 文件"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    print("======grid_size=============>"+str(grid_size))
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    # 获取坐标范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    # 使用 np.histogram2d 统计每个网格的点数量
    counts, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=grid_size,
                                              range=[[x_min, x_max], [y_min, y_max]])
    # 将点数量转换为整数
    counts = counts.astype(int)
    # 统计每个数量对应的网格数目
    grid_distribution = {}
    for count in counts.flatten():
        grid_distribution[count] = grid_distribution.get(count, 0) + 1

    # 构建 grid_info: 每个小格子包含的点索引
    grid_info = {}
    for idx, (x, y) in enumerate(points):
        row = np.searchsorted(x_edges, x, side='right') - 1
        col = np.searchsorted(y_edges, y, side='right') - 1
        if (row, col) not in grid_info:
            grid_info[(row, col)] = []
        grid_info[(row, col)].append(idx)

    return grid_distribution, counts, grid_info


def select_n_samples_multiround_with_prior(data, grid_info, target_total):
    """
    多轮从二维网格中逐步选取数据，基于小格子内数据数量的先验规则。

    参数:
    - data: 输入数据列表，每个元素是字典，包含 'x_tsne', 'y_tsne' 和 'original_index'。
    - grid_info: 小格子分布信息，格式为 {(row, col): [index1, index2, ...]}。
    - target_total: 目标提取的总数量。

    返回:
    - selected_data: 选中的数据列表。
    """
    selected_indices = set()  # 已选数据索引集合
    remaining_grid_info = grid_info.copy()  # 剩余的网格信息
    round_count = 0  # 记录轮数
    total_selected = 0  # 已选数据总数

    while total_selected < target_total and remaining_grid_info:
        round_count += 1
        print(f"Round {round_count}: Starting selection...")

        current_round_indices = set()  # 当前轮次选中的索引
        for (row, col), cell_indices in list(remaining_grid_info.items()):
            cell_size = len(cell_indices)

            # 跳过小格子数据数量 < 200 的格子
            if cell_size < 200:
                del remaining_grid_info[(row, col)]
                continue

            # 根据规则选择数据
            if 200 <= cell_size <= 800:
                # 挑 1 个数据
                to_select = min(1, cell_size)
            else:
                # 挑 2 个数据
                to_select = min(2, cell_size)

            # 更新当前轮次选中的索引
            current_round_indices.update(cell_indices[:to_select])

            # 更新小格子中的剩余数据
            remaining_grid_info[(row, col)] = cell_indices[to_select:]

            # 如果小格子已无数据，移除该格子
            if not remaining_grid_info[(row, col)]:
                del remaining_grid_info[(row, col)]

            # 如果达到目标总数，则终止
            if len(current_round_indices) + total_selected >= target_total:
                break

        # 更新全局已选集合
        selected_indices.update(current_round_indices)
        total_selected = len(selected_indices)

        # 打印轮次进度
        print(f"Round {round_count}: Selected {len(current_round_indices)} points, Total selected: {total_selected}")

        # 如果达到目标总数，提前退出
        if total_selected >= target_total:
            break

    # 根据最终选中的索引提取数据
    selected_data = [data[i] for i in selected_indices]
    return selected_data


# 主函数：从数据统计到多轮选择
if __name__ == "__main__":
    grid_size = 30
    target_total = 20000
    input_file = 'AAAI_set_900w_pipei_304w_with_coords_sort.jsonl'
    output_file = '200_to_0_200_800_1_dayu_800_2/AAAI_set_900w_pipei_304w_with_coords_sort_scalinglaw_2w_30X30.jsonl'
    # 加载数据
    data = load_jsonl(input_file)
    # 提取点的二维坐标
    points = np.array([[item['x_tsne'], item['y_tsne']] for item in data])

    # 分析网格分布
    grid_distribution, counts, grid_info = analyze_grid_distribution(points, grid_size)

    # 多轮挑选数据
    selected_data = select_n_samples_multiround_with_prior(data, grid_info, target_total)
    # 保存到新的 jsonl 文件
    save_jsonl(selected_data, output_file)

    print(f"Successfully saved 50000 samples to {output_file}")

    print(f"Total selected data: {len(selected_data)}")
