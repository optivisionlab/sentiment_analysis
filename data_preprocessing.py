import os
import re
import pandas as pd
import numpy as np
import torch
import pytesseract
from pdf2image import convert_from_path
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
import config

def extract_text_from_pdf(pdf_path):
    """Đọc file PDF, chuyển sang ảnh và dùng Tesseract để trích xuất text."""
    if not os.path.exists(pdf_path):
        return ""
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='vie')
        return text.strip()
    except Exception as e:
        print(f"Lỗi đọc PDF {pdf_path}: {e}")
        return ""

def process_content_ocr(df):
    """Tìm và xử lý OCR nếu cột Content chứa đường dẫn file .pdf"""
    def parse_content(text):
        text = str(text)
        # Nếu dòng đó chứa chuỗi 'content/pdf/...pdf'
        if ".pdf" in text.lower() and "/" in text:
            # Lấy dòng đầu tiên (hoặc xử lý Regex tùy format chuẩn của bạn)
            pdf_rel_path = text.split('\n')[0].strip()
            print(pdf_rel_path)
            return extract_text_from_pdf(pdf_rel_path)
        return text

    print("[*] Đang xử lý Text và OCR PDF...")
    df['Content_Clean'] = df['Content'].apply(parse_content)
    
    # Ghép Title và Content, sau đó phân tách từ tiếng Việt cho PhoBERT
    df['Full_Text'] = df['Title'].astype(str) + " . " + df['Content_Clean']
    print("[*] Đang phân tách từ (Word Segmentation)...")
    df['Full_Text'] = df['Full_Text'].apply(lambda x: ViTokenizer.tokenize(x))
    
    # Tạo nhị phân: >= 7 là tích cực (1), ngược lại (0)
    df['Binary_Target'] = (df['GPT chấm điểm'] >= 7).astype(float)
    return df

def extract_phobert_embeddings(texts, device, batch_size=16):
    """Trích xuất vector [CLS] từ PhoBERT."""
    print(f"[*] Khởi tạo PhoBERT trên {device}...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    model.eval()

    all_embeddings = []
    texts = texts.tolist()

    print(f"[*] Đang mã hóa {len(texts)} bài viết...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=config.MAX_LEN, return_tensors="pt"
            ).to(device)
            
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)

    del model, tokenizer
    torch.cuda.empty_cache()
    return np.array(all_embeddings)