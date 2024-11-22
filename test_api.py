import streamlit as st
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import pandas as pd

# Kết nối đến cơ sở dữ liệu Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))


qdrant_client = QdrantClient(
    url="https://108a5241-04bf-4d9e-9c98-d1deeeefcb3e.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="zpt50bX6nRJoxsj0C9nMQfPtdUL71s2GmJcrYOLvYOlt-5yWBAydzg",
)

model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0)

def find_related_nodes_by_person(tx, identifier, identifier_type):
    """
    Truy vấn tất cả các node liên quan đến người dùng dựa trên identifier và identifier_type.
    identifier_type có thể là 'so_dinh_danh', 'ma_so_bhyt', hoặc 'bien_so'.
    """
    if identifier_type == "so_dinh_danh":
        query = """
        MATCH (p:Person {so_dinh_danh: $identifier})-[r]-(connected)
        RETURN p AS NodeGoc, type(r) AS LoaiQuanHe, connected AS NodeKetNoi
        """
    elif identifier_type == "ma_so_bhyt":
        query = """
        MATCH (p:Person)-[:HAS_INSURANCE_CARD]->(i:InsuranceCard {ma_so_bhyt: $identifier})-[r]-(connected)
        WITH p, type(r) AS LoaiQuanHe, connected
        MATCH (p)-[r]-(related)
        RETURN p AS NodeGoc, LoaiQuanHe, related AS NodeKetNoi
        """
    elif identifier_type == "bien_so":
        query = """
        MATCH (p:Person)-[:HAS_VEHICLE_REGISTRATION]->(v:VehicleRegistration {bien_so: $identifier})-[r]-(connected)
        WITH p, type(r) AS LoaiQuanHe, connected
        MATCH (p)-[r]-(related)
        RETURN p AS NodeGoc, LoaiQuanHe, related AS NodeKetNoi
        """
    else:
        raise ValueError("identifier_type không hợp lệ")

    result = tx.run(query, identifier=identifier)
    node_goc = None
    related_nodes = []
    
    for record in result:
        if node_goc is None:
            node_goc = record["NodeGoc"]
        related_nodes.append({
            "LoaiQuanHe": record["LoaiQuanHe"],
            "NodeKetNoi": record["NodeKetNoi"]
        })
    
    return node_goc, related_nodes

def recognition(path):
    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load the image. Check the file path.")
    faces, _ = mtcnn.detect(img)
    if faces is not None:
        print(len(faces))
        for face in faces:
            bbox = list(map(int,face.tolist()))
            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = np.transpose(face_img, (2, 0, 1))
            face_img = torch.tensor(face_img, dtype=torch.float32)
            face_img = (face_img - 127.5) / 128.0  
            face_img = face_img.unsqueeze(0) 
            embedding = model(face_img).detach()
            embedding_array = embedding.numpy().tolist()

            results = qdrant_client.query_points(
                collection_name="Face_recognition",
                query=embedding_array[0],
                limit=1
            )
            
            first_point = results.points[0]
            score = first_point.score
            id = first_point.id
            if (score<=0.4):
                name = "Unknown"
            else:
                name = first_point.payload['name']
    return id, name
                

# Giao diện Streamlit
st.set_page_config(page_title='Tra cứu thông tin', layout='wide', page_icon=':book:')
st.title("Tìm kiếm thông tin cư dân 360")

# Sử dụng st.columns để chia giao diện thành hai phần
col1, col2 = st.columns([1, 2])

with col1:
    # Nhập số định danh, mã số BHYT, biển số hoặc tải ảnh từ người dùng
    search_type = st.selectbox("Chọn loại tra cứu:", ["Số định danh", "Mã số BHYT", "Biển số", "Hình ảnh"])
    
    so_dinh_danh_value = ""
    ma_so_bhyt_value = ""
    bien_so_value = ""
    uploaded_image = None

    if search_type == "Số định danh":
        so_dinh_danh_value = st.text_input("Nhập số định danh:")
    elif search_type == "Mã số BHYT":
        ma_so_bhyt_value = st.text_input("Nhập mã số BHYT:")
    elif search_type == "Biển số":
        bien_so_value = st.text_input("Nhập biển số xe:")
    elif search_type == "Hình ảnh":
        uploaded_image = st.file_uploader("Tải ảnh lên:", type=["jpg", "jpeg", "png"])

with col2:
    if so_dinh_danh_value:
        identifier_type = "so_dinh_danh"
        identifier = so_dinh_danh_value
    elif ma_so_bhyt_value:
        identifier_type = "ma_so_bhyt"
        identifier = ma_so_bhyt_value
    elif bien_so_value:
        identifier_type = "bien_so"
        identifier = bien_so_value
    elif uploaded_image:
        # Lưu tạm thời ảnh đã tải lên
        temp_image_path = os.path.join("temp_image.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.read())
        
        # Gọi hàm nhận diện
        try:
            id, name = recognition(temp_image_path)
            st.success(f"Nhận diện thành công: {name} (ID: {id})")
            identifier_type = "so_dinh_danh"  # Thay đổi tùy theo cấu trúc Neo4j của bạn
            identifier = str(id)
        except Exception as e:
            st.error(f"Lỗi nhận diện: {e}")
            identifier_type = None
            identifier = None
        finally:
            # Xóa ảnh tạm sau khi xử lý
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    else:
        identifier_type = None
        identifier = None

    # Tìm kiếm trên Neo4j
    if identifier and identifier_type:
        with driver.session(database="Data4life") as session:
            try:
                node, related_nodes = session.execute_read(find_related_nodes_by_person, identifier, identifier_type)
                
                if related_nodes:
                    # Hiển thị thông tin về người trong một expand
                    with st.expander("Thông tin cá nhân"):
                        st.markdown("### Thông tin chi tiết: ")
                        for key, value in node.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    st.write("**Các thông tin liên quan:**")
                    
                    # Gom các node theo label và hiển thị từng nhóm
                    node_groups = {}

                    for item in related_nodes:
                        node_label = list(item["NodeKetNoi"].labels)[0] if item["NodeKetNoi"].labels else "Không có label"
                        
                        if node_label not in node_groups:
                            node_groups[node_label] = []
                        node_groups[node_label].append(item)
                    
                    # Hiển thị các nhóm node theo label
                    for label, items in node_groups.items():
                        with st.expander(f"Nhóm: {label}"):
                            st.write(f"**Số lượng node:** {len(items)}")
                            for index, item in enumerate(items):
                                st.markdown(f"### Thông tin chi tiết của {label} {index}:")
                                for key, value in item["NodeKetNoi"].items():
                                    st.write(f" - **{key}:** {value}")
                                st.write(f"**Loại Quan Hệ:** {item['LoaiQuanHe']}")
                                st.write("-----")
                else:
                    st.warning("Không tìm thấy mối quan hệ nào liên quan.")
            except Exception as e:
                st.error(f"Lỗi khi tìm kiếm trong Neo4j: {e}")
    else:
        if not uploaded_image:
            st.warning("Vui lòng nhập thông tin tìm kiếm hoặc tải ảnh lên.")
# 445948067269