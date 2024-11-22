from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#比赛结束前最后两天售后群发布无水印可视化结果+无标注代码【可直接提交】为了防止倒卖，论文写作过程中遗留数个致命问题，无关代码，该问题解决方式仅在官网授权售后群答疑，盗卖方式购买资料不提供答疑。因倒卖导致无法解决漏洞、赛后无法获奖等 本数模社概不负责  感谢理解 资料助攻购买链接+说明https://docs.qq.com/doc/p/15344cd8ff0275b7d09daa50869317779fd01b25
file_path = 'Data1.xlsx'
china_data = pd.read_excel(file_path, sheet_name='中国')

features = ['中国宠物食品总产值（人民币）', '宠物市场规模 (亿美元)', '宠物食品开支 (亿美元)',
            '兽医服务开支(亿美元)', '中国宠物家庭渗透率', '基尼 (GINI) 系数',
            '总税率（占商业利润的百分比）', '税收（占国民生产总值（GDP）比例）',
            '城镇人口', '农村人口', '食品生产指数（2014-2016 = 100）',
            '人均 GDP（现价美元）', '网络影响力指数']

X = china_data[features]
y_cats = china_data['猫(万)']
y_dogs = china_data['狗(万)']

X = X.dropna()#比赛结束前最后两天售后群发布无水印可视化结果+无标注代码【可直接提交】为了防止倒卖，论文写作过程中遗留数个致命问题，无关代码，该问题解决方式仅在官网授权售后群答疑，盗卖方式购买资料不提供答疑。因倒卖导致无法解决漏洞、赛后无法获奖等 本数模社概不负责  感谢理解 资料助攻购买链接+说明https://docs.qq.com/doc/p/15344cd8ff0275b7d09daa50869317779fd01b25
y_cats = y_cats.loc[X.index]
y_dogs = y_dogs.loc[X.index]

model_cats = RandomForestRegressor(random_state=42)
model_dogs = RandomForestRegressor(random_state=42)

model_cats.fit(X, y_cats)
model_dogs.fit(X, y_dogs)

importance_cats = model_cats.feature_importances_
importance_dogs = model_dogs.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Cats Importance': importance_cats,
    'Dogs Importance': importance_dogs
}).sort_values('Cats Importance', ascending=False)

top_5_cats = importance_df.sort_values('Cats Importance', ascending=False).head(5)#比赛结束前最后两天售后群发布无水印可视化结果+无标注代码【可直接提交】为了防止倒卖，论文写作过程中遗留数个致命问题，无关代码，该问题解决方式仅在官网授权售后群答疑，盗卖方式购买资料不提供答疑。因倒卖导致无法解决漏洞、赛后无法获奖等 本数模社概不负责  感谢理解 资料助攻购买链接+说明https://docs.qq.com/doc/p/15344cd8ff0275b7d09daa50869317779fd01b25
top_5_dogs = importance_df.sort_values('Dogs Importance', ascending=False).head(5)

print(top_5_cats)
print(top_5_dogs)

importance_df = pd.DataFrame({
    'Feature': features,
    'Cats Importance': model_cats.feature_importances_,
    'Dogs Importance': model_dogs.feature_importances_
}).sort_values('Cats Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df.melt(id_vars='Feature', var_name='Type', value_name='Importance'),
            x='Importance', y='Feature', hue='Type')
plt.title('Feature Importance for Cats and Dogs (Random Forest)', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.legend(title='Pet Type')
plt.tight_layout()
plt.show()

# 提取随机森林中的第一棵树
single_tree = model_cats.estimators_[0]

# 方法 1：使用 sklearn 的 plot_tree 可视化
plt.figure(figsize=(20, 10))
plot_tree(single_tree, feature_names=features, filled=True, fontsize=10)
plt.title("Decision Tree from Random Forest (Cats)")
plt.show()

# 方法 2：使用 Graphviz 可视化（生成文件以供查看）
export_graphviz(
    single_tree,
    out_file="../tree.dot",
    feature_names=features,
    filled=True,
    rounded=True,
    special_characters=True
)
