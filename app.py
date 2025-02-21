from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

app = Flask(__name__)

# โหลดโมเดล SBERT ที่รองรับหลายภาษา
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# โหลด CSV
csv_file = "car2.csv"

try:
    df = pd.read_csv(csv_file, encoding="ISO-8859-1")  # หรือ "utf-8" ถ้ามีปัญหา
    print(f"✅ CSV โหลดสำเร็จ! มี {len(df)} แถว")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลด CSV: {e}")
    exit()

# ลบคอลัมน์ที่ไม่จำเป็น
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ตรวจสอบคอลัมน์ที่ต้องใช้
required_columns = {"Manufacturer", "Model", "Price_in_thousands"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV ต้องมีคอลัมน์ {required_columns} แต่พบ {set(df.columns)}")

# รวมข้อมูลที่เกี่ยวข้องเพื่อให้ embedding มีความหมายมากขึ้น
df["combined_info"] = df["Manufacturer"].astype(str) + " | " + df["Model"].astype(str)

# แปลงข้อความเป็นเวกเตอร์ (Embedding)
model_embeddings = np.array([model.encode(text) for text in df["combined_info"]])

# สร้าง FAISS index
index = faiss.IndexFlatL2(model_embeddings.shape[1])
index.add(model_embeddings)

# ฟังก์ชันค้นหาข้อมูลจาก CSV
def retrieve_context(query, top_k=1):  # เปลี่ยน top_k เป็น 1
    query_embedding = model.encode(query).reshape(1, -1)  # ทำให้เป็น 2D array
    distances, indices = index.search(query_embedding, top_k)  # ค้นหาผลลัพธ์

    # ดึงข้อมูลที่เกี่ยวข้องที่สุด
    results = df.iloc[indices[0]]
    car_info = [
        f"🚗 รุ่น: {row['Model']} | ผู้ผลิต: {row['Manufacturer']} | ราคา: {row['Price_in_thousands']} พันดอลลาร์"
        for _, row in results.iterrows()
    ]

    return car_info if car_info else ["⚠️ ไม่พบข้อมูลรถยนต์ที่ตรงกับคำถาม"]

# Route สำหรับหน้าเว็บ
@app.route("/")
def home():
    return render_template("index.html")

# API Endpoint สำหรับรับคำถาม
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"answer": "⚠️ Please enter a question!"})

    # ค้นหาคำตอบ
    context = retrieve_context(user_question)
    response = "\n".join(context)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
