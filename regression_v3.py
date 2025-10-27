import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf


data = {
'y1': [
    1.0785, 1.0896, 1.0807, 1.0803, 1.0866, 1.0617, 1.0714, 1.0672, 1.0718, 1.0643, 
    1.0768, 1.0605, 1.0665, 1.0733, 1.0548, 1.0554, 1.0558, 1.0636, 1.0581, 1.0578, 
    1.0761, 1.0724, 1.0723, 1.0782, 1.0595, 1.0644, 1.0681, 1.0763, 1.0678, 1.0668, 
    1.0625, 1.0686, 1.0699, 1.0738, 1.0657, 1.0798, 1.0872, 1.0825, 1.0760, 1.0724, 
    1.0862, 1.0946, 1.1006, 1.0873, 1.0850, 1.0776, 1.0652, 1.0673, 1.0744
],
'x1': [
    0.5580, 0.7007, 0.8122, 0.9184, 1.0273, 1.1499, 1.3933, 1.2910, 1.3264, 1.3501,
    1.3752, 1.4039, 1.4408, 1.5788, 1.2083, 1.2715, 1.3144, 1.3585, 1.4092, 1.4757,
    1.7230, 1.7383, 1.6110, 1.4727, 1.3929, 1.3388, 1.2982, 1.2806, 1.3328, 1.2999,
    1.2699, 1.2636, 1.2598, 1.2586, 1.2582, 1.3849, 1.3205, 1.2637, 1.2579, 1.2510,
    1.2467, 1.2482, 1.3150, 1.3696, 1.3497, 1.3177, 1.2885, 1.2660, 1.2531
],
'x2': [
    29339,30087,30816,30769,30784,30911,30941,
    30958,30733,31190,30825,30897,31022,30091,
    30915,30984,31326,30971,30537,30546,30297,
    30229,30304,31087,31149,30745,31127,30934,
    30929,30772,31151,30356,30570,30342,30729,
    30726,30307,30736,30490,30411,30346,29927,
    25828,28608,29209,29390,29989,30560,30274
],
'x2_p': [
    3.9147, 3.7823, 3.6884, 3.6516, 3.6569, 3.7129, 3.7890, 3.8198, 3.7847, 3.7631, 
    3.7745, 3.7864, 3.8232, 3.8668, 3.8227, 3.7553, 3.7297, 3.7408, 3.7747, 3.8305, 
    3.9308, 4.3148, 4.3364, 4.2419, 4.0591, 3.9002, 3.7582, 3.6323, 3.7048, 3.7229, 
    3.6816, 3.6109, 3.5949, 3.5542, 3.5339, 3.7835, 3.7914, 3.7201, 3.6010, 3.5683, 
    3.5030, 3.4619, 7.0856, 6.2804, 5.6916, 5.2179, 4.8087, 4.4335, 4.0616
],
'x3': [
    8.5869, 8.8201, 9.0520, 9.1376, 9.1114, 9.0078, 8.6918, 8.6668, 8.7342, 8.7996, 
    8.7762, 8.7392, 8.6688, 8.4541, 8.6839, 8.7812, 8.8606, 8.8406, 8.7716, 8.6413, 
    8.2649, 8.7743, 9.2012, 9.6640, 10.0671, 10.5148, 11.3054, 9.6840, 8.8855, 9.1699, 
    9.4078, 9.6699, 9.8929, 10.1460, 10.5884, 8.2677, 8.7743, 9.2012, 9.6640, 10.0671, 
    10.5148, 11.3054, 6.4540, 6.9880, 7.4164, 7.8341, 8.1573, 8.4093, 8.6360],
'x3_p': [
    2.0999, 2.1247, 2.1491, 2.1584, 2.1539, 2.1409, 2.0992, 2.0989, 2.1055, 2.1129, 
    2.1103, 2.1056, 2.0975, 2.0675, 2.1009, 2.1117, 2.1199, 2.1179, 2.1096, 2.0943, 
    2.0421, 1.1261, 1.4237, 1.6201, 1.7793, 1.9105, 2.0243, 2.1377, 2.0111, 2.0846, 
    2.1214, 2.1523, 2.1737, 2.1942, 2.2218, 1.9125, 2.0443, 2.1104, 2.1650, 2.2036, 
    2.2399, 2.2890, 1.8067, 1.8837, 1.9415, 1.9957, 2.0356, 2.0653, 2.0920],
'alpaca': [
    6.95, 6.31, 9.16, 8.10, 8.66, 8.60, 9.06, 9.62, 7.44, 8.66, 8.16, 7.45, 8.71, 
    8.26, 7.67, 7.52, 9.46, 9.89, 8.06, 9.07, 6.93, 6.38, 5.49, 
    6.67, 7.47, 7.57, 8.39, 8.48, 7.24, 7.57, 7.81, 7.72, 9.73, 9.51, 8.77, 7.49, 
    7.48, 8.69, 7.82, 9.27, 7.93, 8.07, 5.57, 6.42, 6.66, 7.19, 7.76, 10.75, 6.07
]

}
data['x1_l'] = np.log(data['x1'])
data['x22'] = np.array(data['x2']) ** 2
data['x23'] = np.array(data['x2']) ** 3

data['x22_p'] = np.array(data['x2_p']) ** 2
data['x23_p'] = np.array(data['x2_p']) ** 3

data['log_y1'] = np.log(data['y1'])

data['log_alpaca'] = np.log(data['alpaca'])

data['log_x1'] = np.log(data['x1'])
data['log_x2'] = np.log(data['x2'])
data['log_x3'] = np.log(data['x3'])

data['x1_x3'] = np.array(data['x1']) + np.array(data['x3'])
data['x1_x3_p'] = np.array(data['x1']) + np.array(data['x3_p'])
data['x1_x3_p2'] = data['x1_x3_p'] ** 2

data['x1_l_x3'] = np.array(data['x1']) + np.array(data['log_x3'])
data['x1_l_x32'] = data['x1_l_x3'] ** 2

data['l_x1_x3_p'] = np.array(data['log_x1']) - np.array(data['x3_p'])
data['l_x1_l_x3'] = np.array(data['log_x1']) + np.array(data['log_x3'])
data['l_x1_d_l_x3'] = np.array(data['log_x1']) - np.array(data['log_x3'])

data['cate'] = [i for i in range(7) for j in range(7)]
# Convert to DataFrame
df = pd.DataFrame(data)
df['cate'] = df['cate'].astype('category')

# # Prepare independent variables X (add a constant for the intercept)
# X = df[["l_x1_x3_p", "x1_l_x3", "x1_l_x32", "x2_p", "x22_p", "cate", 'y1', 'log_y1', 'alpaca', 'log_alpaca']]
# X = pd.get_dummies(X, columns=['cate'], drop_first=True)

# # Dependent variable Y

# model = smf.ols('y1 ~x1_l_x3 + x22_p + x1_l_x3 * cate_2 + x1_l_x3 * cate_5', data=X).fit() # one useful result
# # 输出回归结果
# print(model.summary())


# Prepare independent variables X (add a constant for the intercept)
X = df[["x1_l", "l_x1_x3_p", 'l_x1_l_x3', "l_x1_d_l_x3", "x1_l_x3", "x1_l_x32", "x2_p", "x22_p", "cate", 'y1', 'log_y1', 'alpaca', 'log_alpaca', 'x2', 'log_x2', 'log_x3']]
X = pd.get_dummies(X, columns=['cate'], drop_first=True)

# Dependent variable Y

model = smf.ols('log_y1 ~x1_l_x3 + log_x2 + x1_l_x3 * cate_2 + cate_5', data=X).fit() # one useful result
# 输出回归结果
print(model.summary())

model = smf.ols('y1 ~l_x1_l_x3 + log_x2', data=X[(X['cate_5']==False) * (X['cate_2']==False) ]).fit() # one useful result
# 输出回归结果
print(model.summary())


model = smf.ols('log_y1 ~l_x1_l_x3 + log_x2 + l_x1_l_x3 * cate_2 + l_x1_l_x3 * cate_5 ', data=X).fit() # one useful result
# 输出回归结果
print(model.summary())


model = smf.ols('log_y1 ~l_x1_x3_p + log_x2 + l_x1_x3_p * cate_2 + l_x1_x3_p * cate_5 ', data=X).fit() # one useful result
# 输出回归结果
print(model.summary())

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 假设df和model是已经定义的变量，进行可视化
plt.figure(figsize=(12, 10))

# 绘制散点图并调整样式
plt.scatter(model.predict(), df['log_y1'], c='darkblue', edgecolor='black', alpha=0.7, s=80, marker='o')

# 设置标题和标签，调整字体大小和样式
plt.xlabel('Predicted Dev Loss', fontsize=32)
plt.ylabel('Loss on Dev Set', fontsize=32)

# 添加网格线
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# 设置背景色
plt.gca().set_facecolor('whitesmoke')
# 保存图像
plt.savefig('log_alpaca_vs_log_y1.png')

# model = smf.ols('log_y1 ~ log_x1 + x3_p + x2_p + x22_p + log_x1 * cate_2 + cate_5', data=X).fit() # SOTA
# model = smf.ols('log_y1 ~ l_x1_l_x3 + x22_p + l_x1_l_x3 * cate_2 + l_x1_l_x3 * cate_5', data=X).fit() # one useful result
# model = smf.ols('y1 ~x1_l_x3 + x22_p + x1_l_x3 * cate_2 + x1_l_x3 * cate_5', data=X).fit() # one useful result


# 创建3D散点图

# def normalize_to_range(arr, new_min=0.2, new_max=0.8):
#     """
#     将数组的值归一化到指定的区间 [new_min, new_max]。

#     :param arr: 输入的 numpy 数组
#     :param new_min: 目标区间的最小值，默认是 0.2
#     :param new_max: 目标区间的最大值，默认是 0.8
#     :return: 归一化后的 numpy 数组
#     """
#     # 获取原数组的最小值和最大值
#     old_min = np.min(arr)
#     old_max = np.max(arr)
    
#     # 进行线性归一化
#     normalized_arr = np.round(np.log( (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min ), 3)
    
#     return normalized_arr


def formula_predict(x1, x2):
    return -0.0113 * x1 + -0.1411 * x2 + 1.5541

y_adjust = -0.1570 * X['cate_2'] - 0.0901 * X['cate_5'] +  0.0594 * X['cate_2'] * X['l_x1_l_x3'] + 0.0400 * X['cate_5'] * X['l_x1_l_x3']
y_adjust = np.array(y_adjust) 

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点，颜色根据y值变化

X1, X2 = np.meshgrid(df['l_x1_l_x3'], df['log_x2'])
Y = -0.0113 * X1 - 0.1411* X2 + 1.5541
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.05)
scatter = ax.scatter(df['l_x1_l_x3'], df['log_x2'], df['log_y1']-y_adjust, c='darkblue', edgecolor='k')
scatter = ax.scatter(df['l_x1_l_x3'], df['log_x2'], formula_predict(df['l_x1_l_x3'], df['log_x2'])+0.0001, c='orange', edgecolor='k')

for i in range(len(df['y1'])):
    error = df['log_y1']-y_adjust - model.predict()[i]
    ax.plot([df['l_x1_l_x3'][i], df['l_x1_l_x3'][i]], [df['log_x2'][i], df['log_x2'][i]], [formula_predict(df['l_x1_l_x3'], df['log_x2'])[i], df['log_y1'][i]-y_adjust[i]], color='black', linestyle='dotted')

# ax.set_xticks([])  # 取消x轴的ticks
#ax.set_yticks([])  # 取消y轴的ticks
# ax.set_zticks([])  # 取消z轴的ticks
# xticks = ax.get_xticks()
# ax.set_xticklabels(normalize_to_range(xticks))  # 将 x 轴的 ticks 修改为它们的平方

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='z', labelsize=20)

# 添加坐标轴标签
ax.set_xlabel('log-Info. Depth', fontsize=28, labelpad=20)
ax.set_ylabel('log-Coverage', fontsize=28, labelpad=20)
ax.set_zlabel('log-Loss on the Dev Set', fontsize=28, labelpad=20)
# ax.set_title('3D Scatter Plot')

# 添加颜色条
# plt.colorbar(scatter, ax=ax, shrink=0.6, label='Y Value')
ax.view_init(elev=30, azim=45)

# 显示图像
plt.savefig('3d.png')


