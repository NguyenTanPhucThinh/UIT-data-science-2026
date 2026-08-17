import os
import re
from typing import Dict, Any, List

class VietnameseSegmenter:
    """Segmenter class supporting VnCoreNLP with PyVi fallback."""
    def __init__(self, use_vncorenlp: bool = True):
        self.use_vncorenlp = use_vncorenlp
        self.rdrsegmenter = None
        self.pyvi_tok = None
        
        if use_vncorenlp:
            try:
                # Ensure JAVA_HOME is set
                if "JAVA_HOME" not in os.environ:
                    java_path = r"C:\Program Files\Java\jre1.8.0_491"
                    if os.path.exists(java_path):
                        os.environ["JAVA_HOME"] = java_path

                import py_vncorenlp
                save_dir = os.path.abspath('models/vncorenlp')
                self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
                print("VietnameseSegmenter: VnCoreNLP initialized successfully.")
            except Exception as e:
                print(f"VietnameseSegmenter Warning: Failed to init VnCoreNLP ({e}). Falling back to PyVi.")
                self.use_vncorenlp = False

        if not self.use_vncorenlp:
            import pyvi.ViTokenizer as pyvi_tok
            self.pyvi_tok = pyvi_tok
            print("VietnameseSegmenter: PyVi initialized as segmenter engine.")

    def segment(self, text: str) -> str:
        """Segments text into compound word units (e.g., bảo_hiểm_y_tế)."""
        if not text:
            return ""
        
        if self.use_vncorenlp and self.rdrsegmenter:
            try:
                # VnCoreNLP word_segment returns list of sentences with segmented words
                segmented_sentences = self.rdrsegmenter.word_segment(text)
                return " ".join(segmented_sentences)
            except Exception as e:
                # Fallback to PyVi on long or special text error
                import pyvi.ViTokenizer as pyvi_tok
                return pyvi_tok.tokenize(text)
        elif self.pyvi_tok:
            return self.pyvi_tok.tokenize(text)
        else:
            return text

def extract_legal_metadata(passage: str, name: str = "", link: str = "") -> Dict[str, Any]:
    """Extract structured legal document metadata (doc_number, issuing_authority, sections)."""
    meta = {
        'doc_number': None,
        'authority': None,
        'num_articles': 0,
        'num_chapters': 0
    }
    
    text = (name or '') + '\n' + (passage or '')
    
    # Match Doc Number (e.g., Số: 5868/QĐ-BYT or Số 15/2022/NĐ-CP)
    match_num = re.search(r'Số(?: hiệu)?:\s*([0-9A-Za-z/._-]+)', text)
    if match_num:
        meta['doc_number'] = match_num.group(1).strip()
        
    # Match Issuing Authority (e.g. BỘ Y TẾ, BỘ TÀI CHÍNH, ỦY BAN NHÂN DÂN...)
    match_auth = re.search(r'^(BỘ\s+[A-ZÀÁẢÃẠÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰYỲÝỶỸỴ\s|-]+|TỔNG CỤC\s+[A-ZÀÁẢÃẠÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰYỲÝỶỸỴ\s|-]+|ỦY BAN NHÂN DÂN\s+[A-ZÀÁẢÃẠÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰYỲÝỶỸỴ\s|-]+)', text, re.MULTILINE)
    if match_auth:
        meta['authority'] = match_auth.group(1).strip()
        
    # Count Articles (Điều ...) and Chapters (Chương ...)
    meta['num_articles'] = len(re.findall(r'\n\s*Điều\s+\d+\.', text))
    meta['num_chapters'] = len(re.findall(r'\n\s*Chương\s+[0-9IVXLCDM]+', text))
    
    return meta
