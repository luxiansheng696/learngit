import re
import jieba
import joblib
import chardet

# 配置
VECTORIZER_PATH = './spam_data/tfidf_vectorizer.pkl'
MODEL_PATH = './spam_data/spam_model.pkl'
STOP_WORDS = set(['的', '了', '在', '是', '我', '有', '和', '就', '不'])

# 加载模型（仅需一次）
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)




def read_email_file(file_path):
    """
    读取邮件文件内容并转为字符串（自动处理编码问题）
    :param file_path: 邮件文件的路径（如 ./test_email.eml、./垃圾邮件.txt）
    :return: 邮件文本字符串（处理失败返回空字符串）
    """
    try:
        # 二进制模式读取，避免编码提前干扰
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        # 自动检测文件编码
        detected_encoding = chardet.detect(raw_data)['encoding']
        # 常见编码兜底（解决chardet检测失败的情况）
        encodings = [detected_encoding, 'gbk', 'utf-8', 'gb2312', 'big5', 'latin-1']

        # 依次尝试编码解码，直到成功
        email_content = ""
        for enc in encodings:
            if enc is None:
                continue
            try:
                email_content = raw_data.decode(enc, errors='strict')
                break
            except (UnicodeDecodeError, LookupError):
                continue

        # 最后兜底：忽略解码错误（保证至少返回可读内容）
        if not email_content:
            email_content = raw_data.decode('gbk', errors='ignore')

        return email_content

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return ""
    except PermissionError:
        print(f"错误：无权限读取文件 {file_path}")
        return ""
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return ""


# 预测函数
def predict_spam(email_text):
    # 预处理
    pattern_header = re.compile(r'^\w+:.+?$', re.MULTILINE)
    pattern_html = re.compile(r'<.*?>', re.S)
    pattern_special = re.compile(r'[^\u4e00-\u9fa5a-zA-Z\s]')

    email_text = pattern_header.sub('', email_text)
    email_text = pattern_html.sub('', email_text)
    email_text = pattern_special.sub(' ', email_text)
    words = jieba.lcut(email_text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    clean_text = ' '.join(words)

    # 预测
    text_vec = vectorizer.transform([clean_text]).toarray()
    pred = model.predict(text_vec)[0]
    return "垃圾邮件" if pred == 1 else "正常邮件"


# 调用（毫秒级响应）
f_path=input('请输入要检测邮件的文件路径（不带引号）：')
f=read_email_file(f_path)
print(predict_spam(f))