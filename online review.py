import re # 正则表达式
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba  # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库
from sklearn.decomposition import PCA # 降维
from sklearn.cluster import KMeans # 聚类


# 获取stopword
def get_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as stopfile:
        stopwords = [line.strip() for line in stopfile.readlines()]
    return set(stopwords)


# 结巴分词
def obj_word(words_filepath, stopwords):
    obj_words = []
    with open(words_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for word in jieba.cut(line.strip().replace(' ','')):
                if word not in stopwords:
                    obj_words.append(word)
    return obj_words


# 词频统计,找到高频词
def word_freqcy(obj_word):
    word_freq = collections.Counter(obj_word) # 对分词作词频统计
    return word_freq


# 打印高频词
def topn_word_freq(word_freq, topn=50):
    word_freq_topn = word_freq.most_common(topn) # 获取前100最高频的词
    print("高频词前50：")
    print(word_freq_topn) # 输出检查


# 词云
def draw_cloud(word_freq):
    mask = np.array(Image.open('wordcloud.png')) # 定义词频背景
    wc = wordcloud.WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
        mask=mask,          # 背景图
        max_words=150,      # 最多显示词数
        max_font_size=100,   # 字体最大值
        background_color='white'
    )
    wc.generate_from_frequencies(word_freq) # 从字典生成词云
    plt.imshow(wc) # 显示词云
    plt.axis('off') # 关闭坐标轴
    plt.show() # 显示图像


# 确定特征值
def feature(word_freq):
    features = []
    for k in word_freq:
         if word_freq[k] <= 100000 and word_freq[k] >= 10:
            features.append(k)
    print("特征值： ")
    print(features)
    return features


# 用频数法为每条评论建立向量
def get_vector_freq(lineword_freq, feature):
    vector = []
    for f in feature:
        if f in lineword_freq:
            vector.append(lineword_freq[f])
        else:
            vector.append(0)
    return vector


# 用独热法为每条评论生成向量
def get_vector_onehot(lineword_freq, feature):
    vector = []
    for f in feature:
        if f in lineword_freq:
            vector.append(1)
        else:
            vector.append(0)
    return vector


# 用权重法
def get_vector_weight(lineword_freq, feature):
    vector = []
    length = np.sum(list(lineword_freq.values()))
    for f in feature:
        if f in lineword_freq:
            vector.append(lineword_freq[f]/length)
        else:
            vector.append(0)
    return vector


# 生成矩阵
def get_vs_list(word_filepath, features, stopwords):
    vs_list = []
    content = {}
    with open(word_filepath, 'r', encoding='utf-8') as file:
        line = file.readline()
        i = 0
        while line:
            content[i] = line
            obj_linewords = []
            for w in jieba.cut(line.strip()):
                if w not in stopwords:
                    obj_linewords.append(w)
            lineword_freq = word_freqcy(obj_linewords)
            v = get_vector_onehot(lineword_freq, features)
            vs_list.append(v)
            line = file.readline()
            i += 1 
    return vs_list, content   


# 计算矩阵的各向量之间的距离
def dist_matrix(vs_list, content):
    length = len(vs_list)
    distance = []
    vector1 = np.array(vs_list)
    for i in range(length):
        vector2 = vs_list[i]
        dis = np.sqrt(np.sum(np.square(vector1 - vector2)))
        distance.append(dis)
        #print("第%d个评论与重心的距离为：%f" %((i+1), distance[i]) )
    pos = distance.index(min(distance))
    print("第%d个评论为评论重心" %(pos+1))
    print("评论内容为：" + content[pos])

# 得到每类聚类中距离重心最近的Num条评论
def near_center_point(review_dict, vector_dict, num=7):
    length = len(review_dict)   
    for i in range(0, length):
        distance = []
        content = review_dict[i]
        vector = vector_dict[i]
        leng = len(review_dict[i])
        for h in range(leng):
            vector2 = vector[h]
            dis = np.sqrt(np.sum(np.square(vector - vector2)))
            distance.append(dis)
        pos = np.argsort(distance)[0:num]
        print(pos)
        print('\n第',(i+1),'类评论，与重心距离最近的',num,'条评论为：')
        print(content[pos])
        

def main():
    stopword = get_stopwords('stopwords_list.txt')
    object_word = obj_word('online_reviews_texts.txt', stopword)
    word_freq = word_freqcy(object_word)
    topn_word_freq(word_freq)
    draw_cloud(word_freq)
    features = feature(word_freq)
    vs_list, content = get_vs_list('online_reviews_texts.txt', features, stopword)
    dist_matrix(vs_list, content)

    # PCA降维
    pca = PCA(n_components=2)
    reduced_vs = pca.fit_transform(vs_list)
    # 绘图
    plt.scatter(reduced_vs[:,0], reduced_vs[:,1], c='y', marker='.')
    plt.show()


    # PCA降维，保留0.8的信息
    pca = PCA(n_components=0.8)
    vs = pca.fit_transform(vs_list)
    # 根据肘部法则，确定簇数
    iter = 30
    clf_inertia = [0.]*iter
    for i in range(1, iter+1, 1):
        clf = KMeans(n_clusters=i, max_iter=300)
        s = clf.fit(vs)
        clf_inertia[i-1] = clf.inertia_
    # 畸变程度曲线
    plt.figure()
    plt.plot(np.linspace(1, iter, iter), clf_inertia, c='b')
    plt.xlabel('center_num')
    plt.ylabel('inertia')
    plt.show()

    # 聚类中心数量为7
    k = 7
    clf = KMeans(n_clusters=k)
    clf.fit(vs)
 
    # 得到每类聚类的评论和向量，画聚类结果图
    review = []
    with open('online_reviews_texts.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        review = np.array(lines)
    vs_list = np.array(vs_list)
    review_dict = {}  # 每类的评论
    vector_dict = {}  # 每类的评论对应的向量
    color = ['r','y','b','g','c','m','k']
    for i in range(k):
        members = clf.labels_ == i
        review_dict[i] = review[members]
        vector_dict[i] = vs_list[members]
        xs = reduced_vs[members, 0]
        ys = reduced_vs[members, 1]
        plt.scatter(xs, ys, c=color[i], marker='.')
    plt.show()
    near_center_point(review_dict, vector_dict, num=7)

if __name__=='__main__':
    main()
