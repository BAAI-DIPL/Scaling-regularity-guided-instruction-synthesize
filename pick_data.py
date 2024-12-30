import numpy as np
import json


def load_jsonl(file_path):
    """加载 jsonl 文件，每行作为一个字典存储到列表中"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """将数据写入 jsonl 文件"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def find_bins_2d(points, grid_size=10):
    """
    将二维点分配到网格中，每个 bin 只放一个点，返回每个 bin 中的点索引列表
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # 获取坐标边界
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 使用 np.histogram2d 划分 bins
    _, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=grid_size,
                                         range=[[x_min, x_max], [y_min, y_max]])

    # 确定每个点落在哪个 bin
    x_bin_indices = np.digitize(points[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(points[:, 1], y_edges) - 1

    num_x_bins = len(x_edges) - 1
    num_y_bins = len(y_edges) - 1
    result = [[[] for _ in range(num_y_bins)] for _ in range(num_x_bins)]
    bin_occupied = [[False for _ in range(num_y_bins)] for _ in range(num_x_bins)]

    for i in range(len(points)):
        x_idx = x_bin_indices[i]
        y_idx = y_bin_indices[i]
        if 0 <= x_idx < num_x_bins and 0 <= y_idx < num_y_bins:
            if not bin_occupied[x_idx][y_idx]:
                result[x_idx][y_idx].append(i)
                bin_occupied[x_idx][y_idx] = True
    return result


# todo: 1w,2w,5w
def select_n_samples_multiround(data, grid_size=100, target_total=1000000):
    """
    按照 bin 的顺序分多轮提取数据，直到达到目标数量
    """
    remaining_data = data.copy()  # 剩余的数据（初始为所有数据）
    selected_indices = set()  # 已选中的数据索引集合
    target_count = target_total  # 目标总数
    round_count = 0  # 记录轮数

    while len(selected_indices) < target_count:
        round_count += 1
        print(f"Round {round_count}: Selecting from remaining {len(remaining_data)} points...")

        # 提取当前剩余数据的坐标
        points = np.array([[item['x_tsne'], item['y_tsne']] for item in remaining_data])

        # 调用 find_bins_2d 获取非空 bins
        bin_sample_ls = find_bins_2d(points, grid_size)

        # 遍历 bins，按顺序收集点
        current_indices = []
        for row in bin_sample_ls:
            for cell in row:
                if cell:  # bin 非空
                    current_indices.append(cell[0])
                    if len(current_indices) + len(selected_indices) >= target_count:
                        break
            if len(current_indices) + len(selected_indices) >= target_count:
                break

        # 将当前轮次的数据索引加入已选集合
        current_selected_indices = {remaining_data[i]['original_index'] for i in current_indices}
        selected_indices.update(current_selected_indices)

        # 打印进度
        print(f"Round {round_count}: Selected {len(current_selected_indices)} points, Total: {len(selected_indices)}")

        # 剔除已选中的数据
        remaining_data = [item for item in remaining_data if item['original_index'] not in selected_indices]

    # 根据最终选中的索引提取数据
    final_selected_data = [data[i] for i in selected_indices]
    return final_selected_data


# 输入和输出文件路径
input_file = 'AAAI_set_900w_pipei_304w_with_coords_sort_base_loss_value_log_third_level_len_sort.jsonl'
output_file = 'AAAI_set_900w_pipei_304w_with_coords_sort_scalinglaw_20w_100X100.jsonl'

# 加载数据
data = load_jsonl(input_file)

# 为每个数据添加索引（方便后续去重和定位）
for idx, item in enumerate(data):
    item['original_index'] = idx

# 选择 50000 条数据
selected_data = select_n_samples_multiround(data, grid_size=100, target_total=200000)

# 保存到新的 jsonl 文件
save_jsonl(selected_data, output_file)

print(f"Successfully saved 50000 samples to {output_file}")
