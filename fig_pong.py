import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats  # 분산/상관 검정용

# 간단 p-value 포맷터 (그래프 타이틀 등에 사용)
def format_p_value(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

# =========================
# Similarity backends
# =========================

def _cosine_similarity_images(imaginary_imgs, real_imgs):
    """픽셀 평탄화 코사인(기존 기능 유지)."""
    if not imaginary_imgs or not real_imgs:
        return 0.0
    L = min(len(imaginary_imgs), len(real_imgs))
    sims = []
    for i in range(L):
        sim_matrix = cosine_similarity(
            [np.array(imaginary_imgs[i]).reshape(-1)],
            [np.array(real_imgs[i]).reshape(-1)]
        )
        sims.append(float(sim_matrix[0, 0]))
    return float(np.mean(sims)) if sims else 0.0

    

# =========================
# Extraction helpers
# =========================

def _get_real_indices(status):
    return [i for i, s in enumerate(status) if s == 0]

def _extract_imag_fragments_with_source(data, real_indices):
    """(fragment_imgs, source_real_idx) 리스트 반환."""
    status = data['status']
    im_imgs = data['im_vectors']
    out = []
    for pos, r_idx in enumerate(real_indices):
        start = r_idx + 1
        end = real_indices[pos + 1] if pos + 1 < len(real_indices) else len(status)
        cur = []
        for t in range(start, end):
            st = status[t]
            if st == 2 and t < len(im_imgs):
                cur.append(im_imgs[t])
            elif st in (1, 3):
                if cur:
                    out.append((cur, r_idx))
                    cur = []
        if cur:
            out.append((cur, r_idx))
    return out

def _extract_future_real_images(data, real_indices, start_real_order, max_length):
    """해당 real order 이후의 real 프레임들 max_length만큼 추출."""
    real_imgs = data['real_vectors']
    out = []
    for k in range(max_length):
        order = start_real_order + k
        if order < len(real_indices):
            r_idx = real_indices[order]
            if r_idx < len(real_imgs):
                out.append(real_imgs[r_idx])
    return out

def _get_next_real_action(data, real_indices, source_order):
    """다음 real step의 action index (없으면 -1)."""
    if source_order + 1 >= len(real_indices):
        return -1
    next_r = real_indices[source_order + 1]
    if next_r >= len(data['tree_reps']['cur_action']):
        return -1
    return int(np.argmax(data['tree_reps']['cur_action'][next_r]))

# =========================
# Main analysis
# =========================

def analyze_imaginary_image_similarity(data, similarity_mode="cosine"):
    """
    각 real step 근처 imagination fragment가 미래 real 이미지들과 얼마나 유사한지 계산.
    폴더 실행 경로에서는 픽셀 평탄화 코사인 유사도만 사용.
    """
    status = data['status']
    real_indices = _get_real_indices(status)
    fragments_with_source = _extract_imag_fragments_with_source(data, real_indices)

    # cosine 유사도만 사용
    sim_fn = _cosine_similarity_images
    fragment_results = []

    # 미리 dict로 real_idx -> order 매핑(중복 탐색 제거)
    real_idx_to_order = {idx: o for o, idx in enumerate(real_indices)}

    for frag_imgs, src_ridx in fragments_with_source:
        if not frag_imgs:
            continue

        src_order = real_idx_to_order[src_ridx]
        label_action = _get_next_real_action(data, real_indices, src_order)

        # fragment 길이만큼 미래 real 이미지 수집
        real_images = _extract_future_real_images(
            data, real_indices, start_real_order=src_order + 1, max_length=len(frag_imgs)
        )

        if not real_images:
            fragment_results.append({
                'fragment_imgs': frag_imgs,
                'source_real_idx': src_ridx,
                'source_real_order': src_order,
                'label_action': label_action,
                'real_images': [],
                'predicted_real_steps': [],
                'similarity': np.nan,
                'fragment_length': len(frag_imgs),
                'real_length': 0
            })
            continue

        similarity = sim_fn(frag_imgs, real_images)

        # 예측 real step 인덱스 기록
        pred_steps = []
        for k in range(len(real_images)):
            if src_order + 1 + k < len(real_indices):
                pred_steps.append(real_indices[src_order + 1 + k])

        fragment_results.append({
            'fragment_imgs': frag_imgs,
            'source_real_idx': src_ridx,
            'source_real_order': src_order,
            'label_action': label_action,
            'real_images': real_images,
            'predicted_real_steps': pred_steps,
            'similarity': similarity,
            'fragment_length': len(frag_imgs),
            'real_length': len(real_images)
        })

    return fragment_results

# =========================
# Visualization
# =========================

 


####################################################

def calculate_internal_diversity(action_sequence, num_actions=6):
    """단일 fragment의 내부 diversity - Shannon Entropy (최대 엔트로피로 정규화)"""
    if not action_sequence:
        return 0.0
    
    action_counts = np.bincount(action_sequence, minlength=num_actions)
    action_probs = action_counts / len(action_sequence)
    entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
    
    # 정규화: 전체 가능한 action 개수 기준으로 최대 엔트로피로 나누기
    max_entropy = np.log2(num_actions)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def calculate_inter_fragment_diversity(fragment_actions_list):
    """여러 fragment 간의 diversity - Jaccard Distance 평균"""
    if len(fragment_actions_list) < 2:
        return 0.0
    
    action_sets = [set(actions) for actions in fragment_actions_list if actions]
    
    jaccard_distances = []
    for i in range(len(action_sets)):
        for j in range(i+1, len(action_sets)):
            intersection = len(action_sets[i] & action_sets[j])
            union = len(action_sets[i] | action_sets[j])
            jaccard_distance = 1 - (intersection / union) if union > 0 else 0
            jaccard_distances.append(jaccard_distance)
    
    return np.mean(jaccard_distances) if jaccard_distances else 0.0

def calculate_real_step_diversity(fragments, num_actions=6):
    """하나의 real step에 대해 하나의 diversity 값 반환"""
    if not fragments:
        return 0.0
    
    if len(fragments) == 1:
        return calculate_internal_diversity(fragments[0], num_actions)
    
    else:
        internal_diversities = [calculate_internal_diversity(f, num_actions) for f in fragments]
        internal_avg = np.mean(internal_diversities)
        inter_diversity = calculate_inter_fragment_diversity(fragments)
        return (internal_avg + inter_diversity) / 2

def analyze_imagination_diversity_by_real_step_old(data):
    """Deprecated: --folder 실행 경로에서 사용되지 않음."""
    print("analyze_imagination_diversity_by_real_step_old is deprecated and unused in --folder mode.")
    # 호환을 위해 최신 구현을 호출합니다.
    return analyze_imagination_diversity_by_real_step(data)

 

############################################################

# GPU 가속화된 벡터 diversity 계산
def _flatten_vectors_list(vectors_list):
    """List of arrays -> 2D float32 array (n, d_flat). Robust per-sample flatten."""
    flat = []
    for v in vectors_list:
        a = np.asarray(v)
        flat.append(a.reshape(-1))
    if not flat:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(flat, axis=0).astype(np.float32)

def _stack_vectors_no_flatten(vectors_list):
    """Stack vectors without flattening dims >1 if possible.
    - Squeezes singleton dims (e.g., (1,D) or (D,1) -> (D,)).
    - If any vector remains >1-D after squeeze, falls back to full flatten.
    """
    squeezed = []
    for v in vectors_list:
        a = np.asarray(v)
        a = np.squeeze(a)
        if a.ndim == 0:
            a = a.reshape(1)
        squeezed.append(a)
    try:
        mat = np.stack(squeezed, axis=0).astype(np.float32)
        # ensure 2D
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        return mat
    except Exception:
        # shape mismatch or still not 1D per sample -> fallback to safe flatten
        return _flatten_vectors_list(vectors_list)


# =========================
# AA.py-compatible diversity logic + pre-pooling helpers (2x2 + 3x3)
# =========================


def aa_calculate_internal_vector_diversity(srn_vectors):
    if len(srn_vectors) < 2:
        return 0.0

    vectors = np.array(srn_vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normalized_vectors = vectors / norms
    n = len(normalized_vectors)
    
    if n <= 50:
        cosine_distances = []
        for i in range(n):
            for j in range(i+1, n):
                cosine_sim = np.dot(normalized_vectors[i], normalized_vectors[j])
                cosine_distance = 1 - cosine_sim
                cosine_distances.append(cosine_distance)
    else:
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        indices = np.triu_indices(n, k=1)
        cosine_similarities = similarity_matrix[indices]
        cosine_distances = (1 - cosine_similarities).tolist()

    variance = np.var(cosine_distances) if cosine_distances else 0.0
    normalized_variance = variance / 4.0
    
    return min(normalized_variance, 1.0)


def aa_calculate_inter_fragment_vector_diversity(fragment_vectors_list):
    if len(fragment_vectors_list) < 2:
        return 0.0

    all_vectors = []
    fragment_labels = []
    for i, vectors in enumerate(fragment_vectors_list):
        if vectors:
            all_vectors.extend(vectors)
            fragment_labels.extend([i] * len(vectors))
    
    if len(all_vectors) < 3:
        return 0.0
    
    all_vectors = np.array(all_vectors)
    fragment_labels = np.array(fragment_labels)
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normalized_vectors = all_vectors / norms

    from sklearn.metrics import silhouette_score
    try:
        silhouette_avg = silhouette_score(normalized_vectors, fragment_labels, metric='cosine')
        normalized_silhouette = (silhouette_avg + 1) / 2
        return max(0.0, min(normalized_silhouette, 1.0))
    except:
        return 0.0


def aa_calculate_real_step_vector_diversity(fragments):
    if not fragments:
        return 0.0
    
    if len(fragments) == 1:
        return aa_calculate_internal_vector_diversity(fragments[0])
    
    else:
        internal_diversities = [aa_calculate_internal_vector_diversity(f) for f in fragments]
        internal_avg = np.mean(internal_diversities)
        inter_diversity = aa_calculate_inter_fragment_vector_diversity(fragments)
        return (internal_avg + inter_diversity) / 2


def aa_analyze_imagination_diversity_by_real_step(data):
    # 최적화: NumPy 배열 변환 및 np.where 사용
    status = np.array(data['status']) if not isinstance(data['status'], np.ndarray) else data['status']
    real_indices = np.where(status == 0)[0]
    
    # 기존과 동일: Real step별 fragment SRN encoding vectors 수집
    real_step_fragments = defaultdict(list)
    im_vectors = data['im_vectors']
    
    for i, real_idx in enumerate(real_indices):
        start = real_idx + 1
        end = real_indices[i + 1] if i + 1 < len(real_indices) else len(status)
        
        current_fragment = []
        for t in range(start, end):
            if status[t] == 2:  # imaginary
                if t < len(im_vectors):
                    srn_vector = im_vectors[t]
                    current_fragment.append(srn_vector)
            elif status[t] in (1, 3):  # reset
                if current_fragment:
                    real_step_fragments[real_idx].append(current_fragment)
                    current_fragment = []
        
        if current_fragment:
            real_step_fragments[real_idx].append(current_fragment)
    
    # 기존과 동일: 각 real step별 diversity 계산
    diversity_by_real_step = {}
    for real_idx, fragments in real_step_fragments.items():
        diversity = aa_calculate_real_step_vector_diversity(fragments)
        diversity_by_real_step[real_idx] = diversity
    
    return diversity_by_real_step

# =========================
# 개별 그래프 생성 및 R² 계산
# =========================

def calculate_r_squared(y_true, y_pred):
    """R² 결정계수 계산"""
    if len(y_true) < 2:
        return 0.0
    ss_res = np.sum((y_true - y_pred)d ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

# =========================
# 폴더 내 npy 파일들 자동 처리 및 step별 통계량 계산
# =========================

def parse_filename(filename):
    """파일명에서 gamename, step, number 추출"""
    import re
    # {gamename}_{step}_{number}.npy 형식 파싱
    pattern = r'(.+)_(\d+e\d+)_(\d+)\.npy'
    match = re.match(pattern, filename)
    if match:
        gamename = match.group(1)
        step_str = match.group(2)
        number = int(match.group(3))
        
        # step 문자열을 숫자로 변환 (예: "1e6" -> 1000000)
        if 'e' in step_str:
            base, exp = step_str.split('e')
            step = int(float(base) * (10 ** int(exp)))
        else:
            step = int(step_str)
            
        return gamename, step, number
    return None, None, None

def process_folder_analysis(folder_path, output_dir="./step_analysis"):
    """폴더 내 npy 파일들을 처리하여 step별 통계량 계산 및 시각화"""
    import os
    import glob
    from collections import defaultdict
    
    os.makedirs(output_dir, exist_ok=True)
    
    # npy 파일들 찾기
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    print(f"발견된 npy 파일 수: {len(npy_files)}")
    
    # 파일들을 gamename, step별로 그룹화
    file_groups = defaultdict(lambda: defaultdict(list))
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        gamename, step, number = parse_filename(filename)
        
        if gamename is not None:
            file_groups[gamename][step].append((file_path, number))
            print(f"파일: {filename} -> {gamename}, step: {step}, number: {number}")
    
    # 각 gamename별로 처리
    for gamename, step_groups in file_groups.items():
        print(f"\n=== {gamename} 처리 중 ===")
        
        # step별로 데이터 수집
        step_data = {}
        
        for step, files in step_groups.items():
            print(f"  Step {step}: {len(files)}개 파일")
            
            # 같은 step의 파일들 처리
            step_correlations = {
                'noop_freq_vs_planning_depth': [],
                'noop_freq_vs_image_similarity': [],
                'real_step_image_sim_vs_planning_depth': [],
                'real_step_image_sim_vs_action_diversity': [],
                'noop_freq_vs_action_diversity': [],
                'planning_depth_vs_action_diversity': [],
                'real_step_image_sim_vs_imagination_diversity': [],
                'noop_freq_vs_imagination_diversity': [],
                'planning_depth_vs_imagination_diversity': []
            }
            
            for file_path, number in files:
                try:
                    print(f"    파일 {number} 처리 중...")
                    data = np.load(file_path, allow_pickle=True).item()
                    
                    # 각 파일에 대해 통계량 계산
                    correlations = calculate_file_correlations(data)
                    
                    # 결과 저장
                    for key, value in correlations.items():
                        if key in step_correlations and value is not None:
                            step_correlations[key].append(value)
                            
                except Exception as e:
                    print(f"    파일 {file_path} 처리 중 오류: {e}")
                    continue
            
            # 같은 step의 파일들에 대해 결합 통계량 계산
            def _combine_fisher_stouffer(values):
                # values: list of {'r','r2','p','n'}
                if not values:
                    return None
                if len(values) == 1:
                    v = values[0]
                    return {'r': v['r'], 'r2': v['r2'], 'p': v['p'], 'n': v.get('n', None), 'count': 1}

                rs = np.array([v['r'] for v in values], dtype=float)
                ps = np.array([v['p'] for v in values], dtype=float)
                ns = np.array([max(int(v.get('n', 0)), 0) for v in values], dtype=float)
                # 유효 항목 필터
                mask = np.isfinite(rs) & np.isfinite(ps) & (ns >= 1)
                if not np.any(mask):
                    return None
                rs = rs[mask]; ps = ps[mask]; ns = ns[mask]
                # Fisher z 결합 (가중치 = n-3, 하한 1)
                rs_clipped = np.clip(rs, -0.999999, 0.999999)
                z = np.arctanh(rs_clipped)
                w = np.maximum(ns - 3.0, 1.0)
                z_bar = np.sum(w * z) / np.sum(w)
                r_bar = np.tanh(z_bar)
                r2_bar = float(r_bar ** 2)
                # Stouffer 방법 (방향성 반영)
                try:
                    z_p = stats.norm.isf(ps / 2.0) * np.sign(rs)
                    Z = np.sum(w * z_p) / np.sqrt(np.sum(w ** 2))
                    p_comb = float(2.0 * stats.norm.sf(abs(Z)))
                except Exception:
                    p_comb = float(np.mean(ps))
                return {'r': float(r_bar), 'r2': r2_bar, 'p': p_comb, 'n': int(np.sum(ns)), 'count': int(np.sum(mask))}

            step_avg_correlations = {}
            for key, values in step_correlations.items():
                if values:
                    step_avg_correlations[key] = _combine_fisher_stouffer(values)
                else:
                    step_avg_correlations[key] = None
            
            step_data[step] = step_avg_correlations
        
        # step별 그래프 생성
        create_step_analysis_plots(gamename, step_data, output_dir)
    
    print(f"\n모든 분석이 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")

def calculate_file_correlations(data):
    """단일 파일에 대해 9개 상관관계 계산"""
    try:
        # 데이터 준비
        fragment_results = analyze_imaginary_image_similarity(data, similarity_mode="cosine")
        # imagination diversity는 AA.py 로직으로 계산해 3개 그래프 및 상관계수에 반영
        diversity_results = aa_analyze_imagination_diversity_by_real_step(data)
        
        # Real step별 데이터 준비
        status = np.array(data['status']) if not isinstance(data['status'], np.ndarray) else data['status']
        real_indices = np.where(status == 0)[0]
        
        if len(real_indices) < 10:  # 데이터가 너무 적으면 None 반환
            return {key: None for key in [
                'noop_freq_vs_planning_depth', 'noop_freq_vs_image_similarity',
                'real_step_image_sim_vs_planning_depth', 'real_step_image_sim_vs_action_diversity',
                'noop_freq_vs_action_diversity', 'planning_depth_vs_action_diversity',
                'real_step_image_sim_vs_imagination_diversity', 'noop_freq_vs_imagination_diversity',
                'planning_depth_vs_imagination_diversity'
            ]}
        
        # Real step별 평균 similarity와 planning depth 계산
        real_step_similarities = defaultdict(list)
        real_step_depths = defaultdict(list)
        
        for result in fragment_results:
            real_idx = result['source_real_idx']
            if np.isfinite(result['similarity']):
                real_step_similarities[real_idx].append(result['similarity'])
            real_step_depths[real_idx].append(result['fragment_length'])
        
        # 평균값 계산
        avg_similarities = []
        avg_depths = []
        diversities = []
        real_step_actions = []
        
        for real_idx in real_indices:
            if real_idx in diversity_results:
                diversities.append(diversity_results[real_idx])
                
                if real_idx in real_step_similarities:
                    avg_similarities.append(np.mean(real_step_similarities[real_idx]))
                else:
                    avg_similarities.append(np.nan)
                
                if real_idx in real_step_depths:
                    avg_depths.append(np.mean(real_step_depths[real_idx]))
                else:
                    avg_depths.append(np.nan)
                
                if real_idx < len(data['tree_reps']['cur_action']):
                    action_onehot = data['tree_reps']['cur_action'][real_idx]
                    action_idx = np.argmax(action_onehot)
                    real_step_actions.append(action_idx)
                else:
                    real_step_actions.append(-1)
        
        # 유효한 데이터만 필터링
        avg_similarities = np.array(avg_similarities)
        avg_depths = np.array(avg_depths)
        diversities = np.array(diversities)
        real_step_actions = np.array(real_step_actions)
        
        valid_mask = np.isfinite(avg_similarities) & np.isfinite(avg_depths) & np.isfinite(diversities)
        valid_similarities = avg_similarities[valid_mask]
        valid_depths = avg_depths[valid_mask]
        valid_diversities = diversities[valid_mask]
        valid_actions = real_step_actions[valid_mask]
        
        # Action diversity 계산 (real_indices 중 diversity가 존재하는 항목과 동일한 순서로 정렬)
        action_diversities_full = calculate_action_diversities(data, real_indices)
        selected_action_diversities = []
        for i, real_idx in enumerate(real_indices):
            if real_idx in diversity_results:
                selected_action_diversities.append(action_diversities_full[i])
        action_diversities = np.array(selected_action_diversities)
        valid_action_diversities = action_diversities[valid_mask]
        
        correlations = {}
        
        # 1. NOOP frequency vs planning depth (sliding window)
        if len(valid_actions) > 160:
            window_size, stride = 160, 2
            window_noop_freq, window_avg_depth = calculate_sliding_window_correlation(
                valid_actions, valid_depths, window_size, stride, lambda x: np.sum(x == 0) / len(x)
            )
            if len(window_noop_freq) > 1:
                r, p = stats.pearsonr(window_noop_freq, window_avg_depth)
                r2 = calculate_r_squared(window_avg_depth, np.poly1d(np.polyfit(window_noop_freq, window_avg_depth, 1))(window_noop_freq))
                correlations['noop_freq_vs_planning_depth'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_noop_freq)}
            else:
                correlations['noop_freq_vs_planning_depth'] = None
        else:
            correlations['noop_freq_vs_planning_depth'] = None
        
        # 2. NOOP frequency vs image similarity (sliding window)
        if len(valid_actions) > 160:
            window_noop_freq, window_avg_sim = calculate_sliding_window_correlation(
                valid_actions, valid_similarities, window_size, stride, lambda x: np.sum(x == 0) / len(x)
            )
            if len(window_noop_freq) > 1:
                r, p = stats.pearsonr(window_noop_freq, window_avg_sim)
                r2 = calculate_r_squared(window_avg_sim, np.poly1d(np.polyfit(window_noop_freq, window_avg_sim, 1))(window_noop_freq))
                correlations['noop_freq_vs_image_similarity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_noop_freq)}
            else:
                correlations['noop_freq_vs_image_similarity'] = None
        else:
            correlations['noop_freq_vs_image_similarity'] = None
        
        # 3. Real step: Image similarity vs planning depth
        if len(valid_similarities) > 1 and len(valid_depths) > 1:
            r, p = stats.pearsonr(valid_similarities, valid_depths)
            r2 = calculate_r_squared(valid_depths, np.poly1d(np.polyfit(valid_similarities, valid_depths, 1))(valid_similarities))
            correlations['real_step_image_sim_vs_planning_depth'] = {'r': r, 'r2': r2, 'p': p, 'n': len(valid_similarities)}
        else:
            correlations['real_step_image_sim_vs_planning_depth'] = None
        
        # 4. Real step: image similarity vs action diversity
        if len(valid_similarities) > 1 and len(valid_action_diversities) > 1:
            r, p = stats.pearsonr(valid_similarities, valid_action_diversities)
            r2 = calculate_r_squared(valid_action_diversities, np.poly1d(np.polyfit(valid_similarities, valid_action_diversities, 1))(valid_similarities))
            correlations['real_step_image_sim_vs_action_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(valid_similarities)}
        else:
            correlations['real_step_image_sim_vs_action_diversity'] = None
        
        # 5. NOOP frequency vs action diversity (sliding window)
        if len(valid_actions) > 160:
            window_noop_freq, window_avg_action_div = calculate_sliding_window_correlation(
                valid_actions, valid_action_diversities, window_size, stride, lambda x: np.sum(x == 0) / len(x)
            )
            if len(window_noop_freq) > 1:
                r, p = stats.pearsonr(window_noop_freq, window_avg_action_div)
                r2 = calculate_r_squared(window_avg_action_div, np.poly1d(np.polyfit(window_noop_freq, window_avg_action_div, 1))(window_noop_freq))
                correlations['noop_freq_vs_action_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_noop_freq)}
            else:
                correlations['noop_freq_vs_action_diversity'] = None
        else:
            correlations['noop_freq_vs_action_diversity'] = None
        
        # 6. Planning depth vs action diversity (sliding window)
        if len(valid_depths) > 160:
            window_avg_depth, window_avg_action_div = calculate_sliding_window_correlation(
                valid_depths, valid_action_diversities, window_size, stride, np.mean
            )
            if len(window_avg_depth) > 1:
                r, p = stats.pearsonr(window_avg_depth, window_avg_action_div)
                r2 = calculate_r_squared(window_avg_action_div, np.poly1d(np.polyfit(window_avg_depth, window_avg_action_div, 1))(window_avg_depth))
                correlations['planning_depth_vs_action_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_avg_depth)}
            else:
                correlations['planning_depth_vs_action_diversity'] = None
        else:
            correlations['planning_depth_vs_action_diversity'] = None
        
        # 7. Real step: image similarity vs imagination diversity
        if len(valid_similarities) > 1 and len(valid_diversities) > 1:
            r, p = stats.pearsonr(valid_similarities, valid_diversities)
            r2 = calculate_r_squared(valid_diversities, np.poly1d(np.polyfit(valid_similarities, valid_diversities, 1))(valid_similarities))
            correlations['real_step_image_sim_vs_imagination_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(valid_similarities)}
        else:
            correlations['real_step_image_sim_vs_imagination_diversity'] = None
        
        # 8. NOOP frequency vs imagination diversity (sliding window)
        if len(valid_actions) > 160:
            window_noop_freq, window_avg_imagination_div = calculate_sliding_window_correlation(
                valid_actions, valid_diversities, window_size, stride, lambda x: np.sum(x == 0) / len(x)
            )
            if len(window_noop_freq) > 1:
                r, p = stats.pearsonr(window_noop_freq, window_avg_imagination_div)
                r2 = calculate_r_squared(window_avg_imagination_div, np.poly1d(np.polyfit(window_noop_freq, window_avg_imagination_div, 1))(window_noop_freq))
                correlations['noop_freq_vs_imagination_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_noop_freq)}
            else:
                correlations['noop_freq_vs_imagination_diversity'] = None
        else:
            correlations['noop_freq_vs_imagination_diversity'] = None
        
        # 9. Planning depth vs imagination diversity (sliding window)
        if len(valid_depths) > 160:
            window_avg_depth, window_avg_imagination_div = calculate_sliding_window_correlation(
                valid_depths, valid_diversities, window_size, stride, np.mean
            )
            if len(window_avg_depth) > 1:
                r, p = stats.pearsonr(window_avg_depth, window_avg_imagination_div)
                r2 = calculate_r_squared(window_avg_imagination_div, np.poly1d(np.polyfit(window_avg_depth, window_avg_imagination_div, 1))(window_avg_depth))
                correlations['planning_depth_vs_imagination_diversity'] = {'r': r, 'r2': r2, 'p': p, 'n': len(window_avg_depth)}
            else:
                correlations['planning_depth_vs_imagination_diversity'] = None
        else:
            correlations['planning_depth_vs_imagination_diversity'] = None
        
        return correlations
        
    except Exception as e:
        print(f"파일 처리 중 오류: {e}")
        return {key: None for key in [
            'noop_freq_vs_planning_depth', 'noop_freq_vs_image_similarity',
            'real_step_image_sim_vs_planning_depth', 'real_step_image_sim_vs_action_diversity',
            'noop_freq_vs_action_diversity', 'planning_depth_vs_action_diversity',
            'real_step_image_sim_vs_imagination_diversity', 'noop_freq_vs_imagination_diversity',
            'planning_depth_vs_imagination_diversity'
        ]}

def calculate_sliding_window_correlation(x_data, y_data, window_size, stride, x_func):
    """슬라이딩 윈도우 상관관계 계산"""
    x_windowed = []
    y_windowed = []
    
    for i in range(0, len(x_data) - window_size + 1, stride):
        x_window = x_data[i:i+window_size]
        y_window = y_data[i:i+window_size]
        
        x_val = x_func(x_window)
        y_val = np.mean(y_window)
        
        x_windowed.append(x_val)
        y_windowed.append(y_val)
    
    return np.array(x_windowed), np.array(y_windowed)

def calculate_action_diversities(data, real_indices):
    """Action diversity (fragment 기반):
    - 단일 fragment: intra(Shannon entropy 정규화)
    - 2개 이상: intra 평균과 inter(Jaccard 거리 평균)의 평균
    반환: real_indices 순서에 맞춘 numpy 배열
    """
    status = data['status']
    # Action 개수 추정
    num_actions = 6
    if 'tree_reps' in data and 'cur_action' in data['tree_reps'] and len(data['tree_reps']['cur_action']) > 0:
        try:
            num_actions = int(data['tree_reps']['cur_action'][0].shape[0])
        except Exception:
            num_actions = 6

    action_diversities = []

    for i, real_idx in enumerate(real_indices):
        start = real_idx + 1
        end = real_indices[i + 1] if i + 1 < len(real_indices) else len(status)

        # fragment 단위로 action 시퀀스 수집
        fragments = []
        current_fragment = []
        for t in range(start, end):
            st = status[t]
            if st == 2:  # imaginary
                if t < len(data['tree_reps']['cur_action']):
                    action_onehot = data['tree_reps']['cur_action'][t]
                    action_idx = int(np.argmax(action_onehot))
                    current_fragment.append(action_idx)
            elif st in (1, 3):  # reset에서 fragment 종료
                if current_fragment:
                    fragments.append(current_fragment)
                    current_fragment = []
        if current_fragment:
            fragments.append(current_fragment)

        # diversity 계산 (정의에 따라)
        if fragments:
            diversity = calculate_real_step_diversity(fragments, num_actions=num_actions)
            action_diversities.append(float(diversity))
        else:
            action_diversities.append(0.0)

    return np.array(action_diversities)

def create_step_analysis_plots(gamename, step_data, output_dir):
    """Step별 분석 결과를 그래프로 생성"""
    
    # 9개 그래프 생성
    plot_configs = [
        ('noop_freq_vs_planning_depth', 'NOOP Frequency vs Planning Depth', 'NOOP Frequency', 'Planning Depth'),
        ('noop_freq_vs_image_similarity', 'NOOP Frequency vs Image Similarity', 'NOOP Frequency', 'Image Similarity'),
        ('real_step_image_sim_vs_planning_depth', 'Image Similarity vs Planning Depth', 'Image Similarity', 'Planning Depth'),
        ('real_step_image_sim_vs_action_diversity', 'Image Similarity vs Action Diversity', 'Image Similarity', 'Action Diversity'),
        ('noop_freq_vs_action_diversity', 'NOOP Frequency vs Action Diversity', 'NOOP Frequency', 'Action Diversity'),
        ('planning_depth_vs_action_diversity', 'Planning Depth vs Action Diversity', 'Planning Depth', 'Action Diversity'),
        ('real_step_image_sim_vs_imagination_diversity', 'Image Similarity vs Imagination Diversity', 'Image Similarity', 'Imagination Diversity'),
        ('noop_freq_vs_imagination_diversity', 'NOOP Frequency vs Imagination Diversity', 'NOOP Frequency', 'Imagination Diversity'),
        ('planning_depth_vs_imagination_diversity', 'Planning Depth vs Imagination Diversity', 'Planning Depth', 'Imagination Diversity')
    ]
    
    for key, title, xlabel, ylabel in plot_configs:
        plt.figure(figsize=(12, 8))
        
        # 데이터 수집
        steps = []
        r_values = []
        r2_values = []
        p_values = []
        counts = []
        
        for step, correlations in sorted(step_data.items()):
            if correlations and key in correlations and correlations[key] is not None:
                steps.append(step)
                r_values.append(correlations[key]['r'])
                r2_values.append(correlations[key]['r2'])
                p_values.append(correlations[key]['p'])
                counts.append(correlations[key]['count'])
        
        if len(steps) > 1:
            # R² 그래프
            plt.subplot(2, 1, 1)
            plt.plot(steps, r2_values, 'o-', linewidth=2, markersize=8, color='blue')
            plt.xlabel('Training Steps')
            plt.ylabel('R²')
            plt.title(f'{gamename}: {title} - R² over Training Steps')
            plt.grid(True, alpha=0.3)
            
            # 파일 개수 표시
            for i, (step, count) in enumerate(zip(steps, counts)):
                plt.annotate(f'n={count}', (step, r2_values[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            # 각 점에 R² 값 표시
            for step, val in zip(steps, r2_values):
                plt.annotate(f'{val:.3f}', (step, val),
                             textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8, color='blue')
            
            # Pearson r 그래프
            plt.subplot(2, 1, 2)
            plt.plot(steps, r_values, 'o-', linewidth=2, markersize=8, color='red')
            plt.xlabel('Training Steps')
            plt.ylabel('Pearson r')
            plt.title(f'{gamename}: {title} - Pearson r over Training Steps')
            plt.grid(True, alpha=0.3)
            
            # p-value 표시
            for i, (step, p_val) in enumerate(zip(steps, p_values)):
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                plt.annotate(significance, (step, r_values[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            # 각 점에 r 값 표시
            for step, val in zip(steps, r_values):
                plt.annotate(f'{val:.3f}', (step, val),
                             textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8, color='red')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{gamename}_{key}_over_steps.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  {title} 그래프 저장 완료")
        else:
            plt.close()
            print(f"  {title}: 데이터 부족으로 그래프 생성 불가")

def main():
    """명령행 인터페이스: 폴더 단위 처리만 수행"""
    parser = argparse.ArgumentParser(description='Pong folder-level analysis')
    parser.add_argument('--folder', required=True, help='폴더 경로 (여러 npy 파일 포함)')
    parser.add_argument('--outdir', default='./step_analysis', help='결과 저장 폴더')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 폴더 단위 처리만 수행
    process_folder_analysis(args.folder, output_dir=args.outdir)

    print(f"완료: 폴더 분석 결과가 '{args.outdir}'에 저장되었습니다.")


if __name__ == '__main__':
    main()
