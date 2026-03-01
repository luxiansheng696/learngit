import chardet
import re
import jieba
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import time

# ========== 配置参数 ==========
path_dataset = '.\\trec06c'
path_dataset_index = '.\\trec06c\\full\\index'
# 持久化文件保存路径（建议创建data文件夹）
SAVE_DIR = './spam_data'
PREPROCESS_DATA_PATH = os.path.join(SAVE_DIR, 'preprocess_data.csv')
VECTORIZER_PATH = os.path.join(SAVE_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(SAVE_DIR, 'spam_model.pkl')
# 中文停用词
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
    '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那'
])


# ========== 工具函数：创建保存目录 ==========
def init_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建保存目录：{SAVE_DIR}")


# ========== 1. 原始数据加载+预处理（仅首次运行） ==========
def load_and_preprocess_raw_data(index_path):
    data = []
    labels = []
    # 预编译正则
    pattern_header = re.compile(r'^\w+:.+?$', re.MULTILINE)
    pattern_html = re.compile(r'<.*?>', re.S)
    pattern_special = re.compile(r'[^\u4e00-\u9fa5a-zA-Z\s]')

    print("首次运行：开始加载并预处理原始邮件数据...")
    start_time = time.time()

    with open(index_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:10000]  # 测试时先取1万条，正式运行去掉[:10000]

    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print(f"已处理 {i}/{len(lines)} 封邮件")

        label, path = line.strip().split()
        full_path = path.replace('..', path_dataset)

        # 读取并解码邮件
        try:
            with open(full_path, 'rb') as mail:
                raw_data = mail.read()
                encodings = ['gbk', 'utf-8', 'gb2312', 'big5']
                content = None
                for enc in encodings:
                    try:
                        content = raw_data.decode(enc, errors='strict')
                        break
                    except:
                        continue
                if content is None:
                    encoding = chardet.detect(raw_data)['encoding']
                    content = raw_data.decode(encoding or 'gbk', errors='ignore')
        except Exception as e:
            print(f"读取邮件失败 {full_path}：{e}")
            content = ""

        # 预处理
        content = pattern_header.sub('', content)
        content = pattern_html.sub('', content)
        content = pattern_special.sub(' ', content)
        words = jieba.lcut(content.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
        clean_content = ' '.join(words)

        if clean_content.strip():
            data.append(clean_content)
            labels.append(1 if label == 'spam' else 0)

    # 保存预处理后的数据（CSV格式，方便查看和复用）
    df = pd.DataFrame({'text': data, 'label': labels})
    df.to_csv(PREPROCESS_DATA_PATH, index=False, encoding='utf-8')
    print(f"预处理完成！耗时：{time.time() - start_time:.2f} 秒")
    print(f"预处理后数据已保存至：{PREPROCESS_DATA_PATH}")
    return data, labels


# ========== 2. 加载预处理好的数据（后续运行） ==========
def load_preprocessed_data():
    print(f"加载已预处理的数据：{PREPROCESS_DATA_PATH}")
    start_time = time.time()
    df = pd.read_csv(PREPROCESS_DATA_PATH, encoding='utf-8')
    data = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"加载完成！耗时：{time.time() - start_time:.2f} 秒")
    return data, labels


# ========== 3. 训练并保存模型（仅首次运行） ==========
def train_and_save_model(X_train, X_test, y_train, y_test):
    print("开始训练模型并保存...")
    start_time = time.time()

    # TF-IDF特征提取
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 训练模型
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    # 保存向量器和模型
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"模型训练并保存完成！耗时：{time.time() - start_time:.2f} 秒")
    print(f"向量器保存至：{VECTORIZER_PATH}")
    print(f"模型保存至：{MODEL_PATH}")

    # 评估模型
    y_pred = model.predict(X_test_vec)
    print("\n模型评估报告：")
    print(classification_report(y_test, y_pred, target_names=['正常邮件', '垃圾邮件']))
    return vectorizer, model


# ========== 4. 加载已保存的模型（后续运行） ==========
def load_saved_model():
    print(f"加载已保存的模型：{MODEL_PATH}")
    start_time = time.time()
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print(f"模型加载完成！耗时：{time.time() - start_time:.2f} 秒")
    return vectorizer, model


# ========== 5. 预测函数（通用） ==========
def predict_spam(email_text, vectorizer, model):
    """预测单条中文邮件是否为垃圾邮件"""
    # 预处理（和训练时保持一致）
    pattern_header = re.compile(r'^\w+:.+?$', re.MULTILINE)
    pattern_html = re.compile(r'<.*?>', re.S)
    pattern_special = re.compile(r'[^\u4e00-\u9fa5a-zA-Z\s]')

    email_text = pattern_header.sub('', email_text)
    email_text = pattern_html.sub('', email_text)
    email_text = pattern_special.sub(' ', email_text)
    words = jieba.lcut(email_text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    clean_text = ' '.join(words)

    # 特征转换+预测
    text_vec = vectorizer.transform([clean_text]).toarray()
    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0][pred]
    return "垃圾邮件" if pred == 1 else "正常邮件", round(prob, 3)


# ========== 主流程（自动判断首次/后续运行） ==========
if __name__ == '__main__':
    init_save_dir()
    total_start = time.time()

    # 第一步：加载数据（优先加载预处理好的）
    if os.path.exists(PREPROCESS_DATA_PATH):
        X, y = load_preprocessed_data()
    else:
        X, y = load_and_preprocess_raw_data(path_dataset_index)

    # 划分训练集/测试集（仅需特征转换时用，预测时不需要）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 第二步：加载/训练模型
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        vectorizer, model = load_saved_model()
    else:
        vectorizer, model = train_and_save_model(X_train, X_test, y_train, y_test)

    # 第三步：测试预测
    print("\n===== 测试预测 =====")
    test_email1 = "尊敬的用户，您的账户已被冻结，请点击链接验证身份，否则将永久封号！"
    test_email2 = "本周工作总结已发送至您的邮箱，麻烦查收并提出修改意见，谢谢。"
    print(f"邮件1：{predict_spam(test_email1, vectorizer, model)}")
    print(f"邮件2：{predict_spam(test_email2, vectorizer, model)}")

    print(f"\n总运行耗时：{time.time() - total_start:.2f} 秒")