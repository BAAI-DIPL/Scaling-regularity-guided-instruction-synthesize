import numpy as np
import json


def load_jsonl(file_path):
    """
    读取 JSONL 文件
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def analyze_grid_distribution(points, grid_size=100):
    """
    分析二维点在网格中的分布情况

    参数:
    - points: 二维点的列表或数组，形状为 (N, 2)，每行是一个点的 (x, y) 坐标。
    - grid_size: 网格划分大小，默认为 100。

    返回:
    - grid_distribution: 网格分布情况，字典形式 {数量: 网格数目}。
    - grid_details: 每个网格中点的数量，二维数组，形状为 (grid_size, grid_size)。
    """
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

    return grid_distribution, counts


def main():
    # 输入文件路径
    input_file = 'AAAI_set_900w_pipei_304w_with_coords_sort.jsonl'

    # 加载数据
    data = load_jsonl(input_file)

    # 提取点的二维坐标
    points = np.array([[item['x_tsne'], item['y_tsne']] for item in data])

    # 分析网格分布
    grid_distribution, grid_details = analyze_grid_distribution(points, grid_size=100)

    # 打印网格分布统计
    print("Grid Distribution:")
    for num_points, num_bins in sorted(grid_distribution.items()):
        print(f"Bins with {num_points} points: {num_bins}")

    # 可视化分布
    import matplotlib.pyplot as plt
    plt.imshow(grid_details, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Point Count')
    plt.title("Point Distribution in Grid")
    plt.show()


if __name__ == "__main__":
    main()
