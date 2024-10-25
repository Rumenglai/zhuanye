import streamlit as st
import time
import base64
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# 自定义页面样式
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #4CAF50;
            font-size: 40px;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #FF5722;
        }
        .success-message {
            font-size: 18px;
            color: #388E3C;
        }
    </style>
""", unsafe_allow_html=True)

# 模拟数据库，用于存储用户的用户名和密码
# 在实际应用中，应该使用数据库存储，并且加密密码
users_db = {'admin': '123456'}

# 页面选择 - 登录、注册或主应用
page = st.sidebar.selectbox(
    "页面导航",
    ["登录", "注册", "主应用"]
)

# 登录页面
if page == "登录":
    st.markdown("<h1 class='main-title'>登录页面</h1>", unsafe_allow_html=True)
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("登录"):
            if username in users_db and users_db[username] == password:
                st.success(f"欢迎回来，{username}！", icon="✅")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
            else:
                st.error("用户名或密码不正确", icon="🚫")

# 注册页面
elif page == "注册":
    st.markdown("<h1 class='main-title'>注册页面</h1>", unsafe_allow_html=True)
    new_username = st.text_input("请输入新用户名")
    new_password = st.text_input("请输入新密码", type="password")
    confirm_password = st.text_input("请确认密码", type="password")

    if st.button("注册"):
        if new_username in users_db:
            st.error("用户名已存在，请选择其他用户名", icon="🚫")
        elif new_password != confirm_password:
            st.error("两次输入的密码不一致", icon="🚫")
        elif new_username == "" or new_password == "":
            st.error("用户名和密码不能为空", icon="🚫")
        else:
            users_db[new_username] = new_password
            st.success("注册成功，请返回登录页面登录", icon="✅")

# 主应用页面
elif page == "主应用":
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.markdown("<h1 class='main-title'>数据处理与分析系统</h1>", unsafe_allow_html=True)

        # 第一个模块：上传数据集，设置对齐率和缺失率
        st.markdown("### 模块 1：上传数据集，设置对齐率和缺失率")
        uploaded_file = st.file_uploader("请选择上传的 Mat 文件", type=['mat'])
        if uploaded_file is not None:
            st.success(f"数据路径：{uploaded_file.name}", icon="📁")

        align_rate = st.slider('数据对齐率', 0, 100, 50)
        st.write(f"当前数据对齐率为：{align_rate}%")

        miss_rate = st.slider('数据缺失率', 0, 100, 50)
        st.write(f"当前数据缺失率为：{miss_rate}%")

        # 第二个模块：选择聚类数量和聚类方法
        st.markdown("### 模块 2：选择聚类数量和聚类方法")
        num_clusters = st.slider('选择聚类的数量', 2, 10, 3)
        cluster_method = st.selectbox('选择聚类方法', ['KMeans', '谱聚类', '层次聚类', 'DBSCAN', '均值漂移'])

        # 第三个模块：开始训练和结束训练
        st.markdown("### 模块 3：训练模型")
        if st.button('开始训练'):
            st.write("正在训练中...")
            progress_bar = st.progress(0)

            # 模拟训练过程
            for i in range(101):
                time.sleep(0.1)  # 模拟训练时间
                progress_bar.progress(i)
                # 每隔20轮次进行一次可视化展示
                if i % 20 == 0 and i != 0:
                    # 创建可视化展示，包含四个子图
                    fig = plt.figure(figsize=(24, 6))
                    for j in range(4):
                        ax = fig.add_subplot(1, 4, j + 1)
                        # 生成模拟数据集，总共10000个数据点，具有2列特征
                        data, _ = make_blobs(n_samples=1000, centers=num_clusters, n_features=2, random_state=42)
                        data = pd.DataFrame(data, columns=['特征 1', '特征 2'])
                        labels = np.random.randint(0, 10, 1000)  # 模拟聚类结果

                        # 子图：聚类结果
                        ax.scatter(data['特征 1'], data['特征 2'], c=labels, cmap='tab10', alpha=0.6, s=10)
                        ax.set_title(f'训练第 {i} 轮 - 子图 {j + 1}', fontsize=10, fontweight='bold')
                        ax.axis('off')

                    # 显示图像
                    st.pyplot(fig)
            st.success("训练完成！", icon="✅")

        if st.button('结束训练'):
            st.warning("训练已结束", icon="⚠️")

        # 第四个模块：展示聚类准确度和损失图
        st.markdown("### 模块 4：聚类准确度与损失展示")
        if st.button('聚类准确度与损失曲线'):
            # 聚类准确度与损失图代码
            epochs = np.arange(1, 151)
            acc = np.random.uniform(0.2, 0.8, len(epochs))
            nmi = np.random.uniform(0.3, 0.8, len(epochs))
            ari = np.random.uniform(0.2, 0.7, len(epochs))
            loss = np.random.uniform(0.1, 8.0, len(epochs))

            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.plot(epochs, acc, 'orange', label='ACC')
            ax1.plot(epochs, nmi, 'green', label='NMI')
            ax1.plot(epochs, ari, 'blue', label='ARI')
            ax1.set_xlabel('Epoch', fontsize=14)
            ax1.set_ylabel('Clustering Performance', fontsize=14)
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2 = ax1.twinx()
            ax2.plot(epochs, loss, 'red', label='Loss')
            ax2.set_ylabel('Loss', fontsize=14)
            ax2.legend(loc='upper right')
            st.image('a.png', caption='Convergence analysis of clustering performance and loss values.',
                     use_column_width=True)

        # 第五个模块：可视化展示

        st.markdown("### 模块 5：聚类可视化")
        if st.button('数据聚类'):
            # 生成模拟数据集，总共10000个数据点，具有10列特征
            np.random.seed(42)
            data = pd.DataFrame(np.random.rand(5000, 10), columns=[f'特征 {i + 1}' for i in range(10)])

            # 根据用户选择的聚类方法进行聚类分析
            if cluster_method == 'KMeans':
                model = KMeans(n_clusters=num_clusters)
            elif cluster_method == '谱聚类':
                from sklearn.cluster import SpectralClustering

                model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
            elif cluster_method == '层次聚类':
                from sklearn.cluster import AgglomerativeClustering

                model = AgglomerativeClustering(n_clusters=num_clusters)
            elif cluster_method == 'DBSCAN':
                from sklearn.cluster import DBSCAN

                model = DBSCAN(eps=0.1)
            elif cluster_method == '均值漂移':
                from sklearn.cluster import MeanShift

                model = MeanShift()

            model.fit(data)
            labels = model.labels_

            # 将聚类结果可视化
            # 创建两个子图，左侧为原始数据，右侧为聚类结果
            # fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            # for j in range(4):
            #     # 生成模拟数据集，总共10000个数据点，具有2列特征
            #     data = pd.DataFrame(np.random.rand(10000, 2), columns=['特征 1', '特征 2'])
            #     labels = np.random.randint(0, 10, 10000)  # 模拟聚类结果
            #
            #     # 子图：聚类结果
            #     axes[j].scatter(data['特征 1'], data['特征 2'], c=labels, cmap='tab10', alpha=0.6, s=10)
            #     axes[j].set_title(f'聚类结果 - 子图 {j + 1}', fontsize=10, fontweight='bold')
            #     axes[j].axis('off')
            #
            # # 显示图像
            # st.pyplot(fig)
            # st.success("聚类分析完成！", icon="✅")
            # 将聚类结果可视化
            # 将聚类结果可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data['特征 1'], y=data['特征 2'], hue=labels, palette='Set1', alpha=0.7, s=60, ax=ax)
            ax.set_title('聚类结果', fontsize=14, fontweight='bold')
            ax.set_xlabel('特征 1')
            ax.set_ylabel('特征 2')
            ax.legend(title='类别')
            st.pyplot(fig)
            st.success("聚类分析完成！", icon="✅")
    else:
        st.warning("请先登录以访问主应用。", icon="⚠️")
