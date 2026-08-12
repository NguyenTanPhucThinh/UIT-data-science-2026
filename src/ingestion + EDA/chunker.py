import re

# Estimate tokens (conservative limit, ~350 words fits into 512 tokens comfortably)
MAX_WORDS_PER_CHUNK = 350
OVERLAP_WORDS = 50

def is_semantic_boundary(line: str) -> bool:
    """Check if a line is a structural boundary in Vietnamese legal text."""
    line_upper = line.strip().upper()
    if line_upper.startswith("ĐIỀU ") or line_upper.startswith("CHƯƠNG ") or line_upper.startswith("MỤC "):
        return True
    return False

def count_words(text: str) -> int:
    return len(text.split())

def sliding_window_chunk(lines: list, segmented_lines: list, max_words=MAX_WORDS_PER_CHUNK, overlap_words=OVERLAP_WORDS):
    """Fallback sliding window chunker when semantic boundaries are too large.
    
    Handles edge cases:
    - Single lines that exceed max_words are split on word boundaries.
    - Empty lines are preserved but don't count toward word limits.
    """
    # First, break any oversized lines into sub-lines by word count
    expanded_lines = []
    expanded_seg = []
    for line, seg_line in zip(lines, segmented_lines):
        line_wc = count_words(line)
        if line_wc > max_words:
            # Split long line into word-level sub-lines
            words = line.split()
            seg_words = seg_line.split()
            for start in range(0, len(words), max_words - overlap_words):
                end = min(start + max_words, len(words))
                expanded_lines.append(' '.join(words[start:end]))
                seg_end = min(start + max_words, len(seg_words))
                expanded_seg.append(' '.join(seg_words[start:seg_end]))
                if end >= len(words):
                    break
        else:
            expanded_lines.append(line)
            expanded_seg.append(seg_line)
    
    chunks = []
    current_chunk_lines = []
    current_seg_lines = []
    current_word_count = 0
    
    for line, seg_line in zip(expanded_lines, expanded_seg):
        line_wc = count_words(line)
        if current_word_count + line_wc > max_words and current_chunk_lines:
            # Yield current chunk
            chunks.append({
                'passage': "\n".join(current_chunk_lines),
                'passage_segmented': "\n".join(current_seg_lines)
            })
            # Overlap: keep last few lines that fit in overlap
            overlap_lines = []
            overlap_seg = []
            overlap_wc = 0
            for r_line, r_seg in zip(reversed(current_chunk_lines), reversed(current_seg_lines)):
                if overlap_wc + count_words(r_line) <= overlap_words:
                    overlap_lines.insert(0, r_line)
                    overlap_seg.insert(0, r_seg)
                    overlap_wc += count_words(r_line)
                else:
                    break
            current_chunk_lines = overlap_lines
            current_seg_lines = overlap_seg
            current_word_count = overlap_wc
            
        current_chunk_lines.append(line)
        current_seg_lines.append(seg_line)
        current_word_count += line_wc
        
    if current_chunk_lines:
        chunks.append({
            'passage': "\n".join(current_chunk_lines),
            'passage_segmented': "\n".join(current_seg_lines)
        })
    return chunks


def chunk_document(doc: dict, max_words=MAX_WORDS_PER_CHUNK, overlap=OVERLAP_WORDS) -> list:
    """
    Chunks a document using structural boundaries (Điều/Chương).
    Matches passage and passage_segmented line-by-line.
    """
    raw_passage = doc.get('passage', '')
    seg_passage = doc.get('passage_segmented', '')
    
    lines = raw_passage.split('\n')
    seg_lines = seg_passage.split('\n')
    
    # Safety fallback if line counts mismatch (should not happen if processed correctly)
    if len(lines) != len(seg_lines):
        return sliding_window_chunk([raw_passage], [seg_passage], max_words, overlap)
        
    # Group lines by semantic sections
    sections = []
    current_section_lines = []
    current_section_seg = []
    current_title = "Phần mở đầu"
    
    for line, seg_line in zip(lines, seg_lines):
        if is_semantic_boundary(line):
            if current_section_lines:
                sections.append((current_title, current_section_lines, current_section_seg))
            current_title = line.strip()
            current_section_lines = [line]
            current_section_seg = [seg_line]
        else:
            current_section_lines.append(line)
            current_section_seg.append(seg_line)
            
    if current_section_lines:
        sections.append((current_title, current_section_lines, current_section_seg))
        
    # Now process sections into chunks constrained by max_words
    final_chunks = []
    
    for title, s_lines, s_seg_lines in sections:
        section_text = "\n".join(s_lines)
        if count_words(section_text) <= max_words:
            final_chunks.append({
                'legal_section': title[:100],
                'passage': section_text,
                'passage_segmented': "\n".join(s_seg_lines)
            })
        else:
            # Section is too large, use sliding window on its lines
            sub_chunks = sliding_window_chunk(s_lines, s_seg_lines, max_words, overlap)
            for sub in sub_chunks:
                sub['legal_section'] = title[:100]
                final_chunks.append(sub)
                
    # Format the final output chunks
    doc_id = doc['doc_id']
    formatted = []
    for i, c in enumerate(final_chunks):
        if not c['passage'].strip():
            continue  # skip empty chunks
        formatted.append({
            'chunk_id': f"{doc_id}_chunk_{len(formatted):03d}",
            'doc_id': doc_id,
            'doc_name': doc.get('name'),
            'doc_type': doc.get('doc_type', 'Khác'),
            'chunk_index': len(formatted),
            'total_chunks': 0, # Placeholder, will be updated below
            'legal_section': c.get('legal_section', ''),
            'passage': c['passage'],
            'passage_segmented': c['passage_segmented']
        })
        
    # Update total chunks now that we know the final count
    total_valid = len(formatted)
    for c in formatted:
        c['total_chunks'] = total_valid
        
    return formatted
