from flask import Flask, jsonify, request, render_template
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import torch
import torch.nn as nn

# Khởi tạo Flask app
app = Flask(__name__)

# Hàm tiền xử lý với underthesea
def preprocess_text(text):
    # Sử dụng underthesea để phân tách từ
    return ' '.join(word_tokenize(text))

# Định nghĩa kiến trúc mô hình
class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.fc = nn.Linear(256, 1)  # Thay đổi kích thước phù hợp với mô hình của bạn

    def forward(self, x):
        return self.fc(x)

# Tải mô hình đã huấn luyện (model.pt)
device = torch.device('cpu')
model = ModelClass()
model.load_state_dict(torch.load('model.pt', map_location=device))  # Tải trọng số
model.eval()  # Đặt mô hình ở chế độ đánh giá

# Tải dữ liệu từ file jobs.json với mã hóa UTF-8
with open('jobs.json', 'r', encoding='utf-8') as f:
    jobs_data = json.load(f)

# Tiền xử lý các mô tả công việc
job_descriptions = [preprocess_text(job['description']) for job in jobs_data]

# Khởi tạo TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(job_descriptions)

# Route trang chủ để render file HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render file index.html từ thư mục templates

# Route nhận yêu cầu tìm kiếm
@app.route('/search', methods=['POST'])
def search():
    # Lấy truy vấn từ người dùng
    query = request.json.get('query')
    
    # Tiền xử lý truy vấn
    query_tfidf = vectorizer.transform([preprocess_text(query)])
    
    # Tính toán độ tương đồng cosine giữa truy vấn và các mô tả công việc
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Lấy 5 công việc có độ tương đồng cao nhất
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    top_jobs = [jobs_data[i] for i in top_indices]

    # Trả kết quả dưới dạng JSON
    return jsonify({'results': top_jobs})

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(port=5000, debug=True)
