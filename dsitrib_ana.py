import os
import sys
sys.path.append('/share/duli/utils')
sys.path.append('/share/project/duli/content_relation_ana/utils')
from utils import *
from draw import distrib_ana_tsne
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, rgb2hex
import matplotlib.pyplot as plt
from FlagEmbedding import FlagModel
import numpy as np
from tqdm import tqdm
import random
import pdb
import re

def find_bins_2d(points, grid_size=10):
    """
    根据二维点的坐标，确定每个点所在的二维 bin，并返回一个存储每个 bin 包含的点索引的列表。
    
    参数:
    points : numpy.ndarray, 形状为 (N, 2)，表示 N 个二维点
    xedges : numpy.ndarray, X 轴的 bin 边界
    yedges : numpy.ndarray, Y 轴的 bin 边界
    
    返回:
    result : list of lists, 每个列表对应一个 bin，存储的是该 bin 中点的索引
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Get the min and max for each axis
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Create a 2D histogram (grid)
    histogram, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=grid_size, 
                                                 range=[[x_min, x_max], [y_min, y_max]])
    

    # 获取点的数量
    num_points = points.shape[0]
    
    # 使用 np.digitize 确定每个点在 x 和 y 方向上分别落在哪个 bin 内
    x_bin_indices = np.digitize(points[:, 0], x_edges) - 1  # 对 X 轴进行 bin 分配
    y_bin_indices = np.digitize(points[:, 1], y_edges) - 1  # 对 Y 轴进行 bin 分配
    
    # 创建一个二维列表，存储每个 bin 中的点索引
    num_x_bins = len(x_edges) - 1  # X 方向的 bin 数量
    num_y_bins = len(y_edges) - 1  # Y 方向的 bin 数量
    
    result = [[[] for _ in range(num_y_bins)] for _ in range(num_x_bins)]
    
    # 遍历每个点，将其索引放入对应的 bin 中
    for i in range(num_points):
        x_idx = x_bin_indices[i]
        y_idx = y_bin_indices[i]
        
        # 检查 x_idx 和 y_idx 是否在有效的 bin 范围内
        if 0 <= x_idx < num_x_bins and 0 <= y_idx < num_y_bins:
            result[x_idx][y_idx].append(i)
    
    return result

def remove_adjacent_duplicates(lst):
    if not lst:
        return []
    result = [lst[0]]
    for num in lst[1:]:
        if num != result[-1]:
            result.append(num)
    return result

def generate_colors(n):
    """
    生成任意数量的颜色。
    
    参数:
    n (int): 需要生成的颜色数量。
    
    返回:
    list: 颜色列表，包含n个颜色。
    """
    colors = plt.cm.hsv(np.linspace(0, 1, n))
    return [rgb2hex(c) for c in colors]

def create_listed_colormap(n):
    """
    创建一个包含灰色，浅蓝，浅绿，红色的 ListedColormap。
    
    参数:
    n (int): 需要生成的颜色数量。
    
    返回:
    ListedColormap: 自定义的颜色映射。
    """
    # 定义固定的颜色列表
    colors = ['#808080',  # 灰色
              '#add8e6',  # 浅蓝色
              '#90ee90',  # 浅绿色
              '#ff0000']  # 红色
    
    # 如果需要的颜色数量超过4个，可以循环使用这些颜色
    if n > 4:
        colors = (colors * ((n // 4) + 1))[:n]
    
    return ListedColormap(colors)

def distrib_ana_tsne(data_tsne, cate_ids, cate_names=None, out_path='./tsne.png', s=0.1, continue_draw=False, alpha=0.9):
    # assert len(cate_ids) == len(cate_names)
    if not cate_names:
        cate_names = remove_adjacent_duplicates(cate_ids)
    cmap1 = plt.get_cmap('tab20b')
    cmap2 = plt.get_cmap('tab20c')
    NUM_CATES = len(set(cate_ids))
    new_cmap = create_listed_colormap(NUM_CATES)
    plt.figure(figsize=(15,9), dpi=300)  # Make the plot flatter
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.2)
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=cate_ids, cmap=new_cmap, marker='o', s=s, alpha=alpha)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([i-float(i)/NUM_CATES for i in range(NUM_CATES)])
    cbar.set_ticklabels(cate_names)
    cbar.set_label('Cluster Label')
    plt.title('DBSCAN Clustering Over t-SNE')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    # plt.savefig('./tsne_llmsys_llmsysseed_1m_152k.png')
    if not continue_draw:
        plt.savefig(out_path)


random.seed(2024)
os.chdir('/share/project/duli/content_relation_ana/subset_gen')
tot_sample_ls = load_jsonl('/share/project/duli/space_entropy_smapling/space_distrib/data_pools/AAAI_set_900w_pipei_304w_with_coords_sort_base_loss_value_log_third_level_len_sort.jsonl')
subset_path = '/share/project/duli/space_entropy_smapling/space_distrib/data_pools'
subset_ls = os.listdir(subset_path)
deita_files = {}
random_files = {}
scaling_files = {}

sizes = ['2w', '10w', '20w', '50w']

for file in subset_ls:
    for size in sizes:
        if re.match(r'.*deita.*\.jsonl', file) and size in file:
            deita_files[size] = load_jsonl(os.path.join(subset_path, file))[0]
        elif re.match(r'.*random.*\.jsonl', file) and size in file:
            random_files[size] = load_jsonl(os.path.join(subset_path, file))[0]
        elif re.match(r'.*scaling.*\.jsonl', file) and size in file:
            scaling_files[size] = load_jsonl(os.path.join(subset_path, file))[0]


sample_ratio = 0.02

model = FlagModel('/share/project/models/bge', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
tot_sample_ls_tmp = random.sample(tot_sample_ls, int(len(tot_sample_ls) * sample_ratio))

for size in sizes:

    deita_file_tmp = random.sample(deita_files[size], int(len(deita_files[size]) * sample_ratio * 2))
    random_file_tmp = random.sample(random_files[size], int(len(random_files[size]) * sample_ratio * 2))
    scaling_file_tmp = random.sample(scaling_files[size], int(len(scaling_files[size]) * sample_ratio * 2))

    subset_sample_ls = tot_sample_ls_tmp + deita_file_tmp + random_file_tmp + scaling_file_tmp

    text_ls = [str(i['content']) if 'content' in i else str(i['input_key']) for i in subset_sample_ls ]
    repre_vectors = model.encode( text_ls ) # 每个元素是一个字符串
    cate_ids = [0] * len(tot_sample_ls_tmp) + [1] * len(deita_file_tmp) + [2] * len(random_file_tmp) + [3] * len(scaling_file_tmp)
    # cate_names = ['Whole Pool'] * len(tot_sample_ls_tmp) + ['Deita'] * len(deita_file_tmp) + ['Random'] * len(random_file_tmp) + ['Scaling'] * len(scaling_file_tmp) 
    cate_names = ['Whole Pool', 'Deita', 'Random', 'Scaling'] 

    # indicate the group of the sample
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(repre_vectors)

    distrib_ana_tsne(data_tsne, s=20,cate_ids =cate_ids, cate_names=cate_names, out_path=f'/share/project/duli/space_entropy_smapling/space_distrib/distrb_{size}.png',alpha=0.1)


tot_sample_ls_tmp = random.sample(tot_sample_ls, 100000)
text_ls = [str(i['content']) if 'content' in i else str(i['input_key']) for i in tot_sample_ls_tmp ]
repre_vectors = model.encode( text_ls ) # 每个元素是一个字符串

tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(repre_vectors)

NUM_GRID = 25
bin_sample_ls = find_bins_2d(data_tsne, NUM_GRID)

mean_loss_ls = []
grid_len_ls = []
for ith, i in enumerate(bin_sample_ls):
    for j in i:
        if len(j) > 0:
            mean_loss_tmp = np.nanmean(sorted([tot_sample_ls_tmp[k]['base_loss_value'] for k in j], reverse=True)[:10])
        else:
            mean_loss_tmp = 0
        grid_len_ls.append(len(j))
        mean_loss_ls.append(mean_loss_tmp)

grid_len_ls = np.array(grid_len_ls).reshape(NUM_GRID, NUM_GRID)
mean_loss_ls = np.array(mean_loss_ls).reshape(NUM_GRID, NUM_GRID)

# draw the heatmap of the mean loss

plt.figure(figsize=(15,9), dpi=300)  # Make the plot flatter
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.2)
plt.imshow(mean_loss_ls, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Mean Loss of Each Bin')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.savefig(f'/share/project/duli/space_entropy_smapling/space_distrib/mean_loss_heatmap_{str(NUM_GRID)}.png')


plt.figure(figsize=(15,9), dpi=300)  # Make the plot flatter
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.2)
plt.imshow(grid_len_ls, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Mean Loss of Each Bin')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.savefig(f'/share/project/duli/space_entropy_smapling/space_distrib/grid_len_heatmap_{str(NUM_GRID)}.png')

# draw the histogram of the grid length
plt.figure(figsize=(15,9), dpi=300)  # Make the plot flatter
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.2)
freq_ls = grid_len_ls.flatten()[grid_len_ls.flatten()>0]
# draw histogram with the log-scale y-axis
plt.hist(freq_ls, bins=100, log=True)
plt.title('Histogram of Grid Length')
plt.xlabel('Grid Length')
plt.ylabel('Frequency')
plt.savefig(f'/share/project/duli/space_entropy_smapling/space_distrib/grid_len_hist_{str(NUM_GRID)}.png')
