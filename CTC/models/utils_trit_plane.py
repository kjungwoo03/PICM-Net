from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ans import BufferedRansEncoder, RansDecoder
from PIL import Image

import torch
import scipy
import math
import torch.nn.functional as F

multiplier = -scipy.stats.norm.ppf(1e-09 / 2)
mode = 3
opt_pnum = 5
pnum_btw_trit = 48
pnum_part = 1.0


__all__ = [
    "opt_pnum",
    "mode",
    "get_Nary_tensor",
    "make_pmf_table",
    "get_ans",
    "TP_entropy_encoding",
    "get_transmission_tensor",
    "TP_entropy_encoding_scalable",
    "get_empty_Nary_tensor",
    "TP_entropy_decoding",
    "TPED_last_point",
    "TPED",
    "prepare_TPED_scalable",
    "get_transmission_tensor_sigma",
    "select_sub_interval_light",
    "TP_entropy_decoding_light",
    "TPED_light",
    "TPED_last_point_light",
    "make_pmf_table_light",
    "TP_entropy_encoding_light",
    "TP_entropy_encoding_scalable_light",
    "select_sub_interval",
    "reconstruct_y_hat_from_digits",
    # "build_sort_index",
    # "TP_entropy_encoding_scalable_ordered",
    # "prepare_TPED_scalable_ordered",
    # "TPED_ordered",
    # "TPED_last_point_ordered",
    "get_transmission_tensor_reverse",
    "get_transmission_tensor_random",
    "get_transmission_tensor_optimal_channel",
    "get_transmission_tensor_optimal_patch"

]

def _ensure_label_tensor(label, device):
    """
    label을 CE용 target 텐서(int64, shape=[1])로 정규화
    """
    if torch.is_tensor(label):
        tgt = label.to(device=device, dtype=torch.long)
        if tgt.ndim == 0:
            tgt = tgt.view(1)
        elif tgt.ndim > 1:
            tgt = tgt.view(-1)[:1]
        return tgt
    else:
        # int, numpy.int 등
        return torch.tensor([int(label)], device=device, dtype=torch.long)

def _symbol_entropies_from_pmfs_norm(pmfs_norm_list):
    """
    pmfs_norm_list: 길이 i+1 리스트, 각 항 shape = [num_symbols_at_level_j, mode]
    반환: concat 순서 기준의 H(p) 벡터 (길이 total_symbols)
    """
    H_list = []
    for p in pmfs_norm_list:
        p_ = p.clamp_min(1e-12)                     # 안정성
        H = -(p_ * p_.log2()).sum(dim=1)            # [num_symbols_at_level_j]
        H_list.append(H)
    return torch.cat(H_list, dim=0)  

def crop_with_pads(x_pad: torch.Tensor, pads):
    # x_pad: (B, C, H2, W2), pads=(l, r, t, b)
    l, r, t, b = pads
    return x_pad[..., t:x_pad.size(-2)-b, l:x_pad.size(-1)-r]

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: (1, C, H, W) or (C, H, W), range [0,1]
    if x.dim() == 4:
        x = x.squeeze(0)
    x = (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
    c, h, w = x.shape
    if c == 1:
        return Image.fromarray(x.squeeze(0).numpy(), mode="L")
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")
    
def _scores_from_order(order_idx: torch.Tensor):
    """
    order_idx: 길이 S, 우리가 원하는 최종 순서(0번째가 가장 먼저 인코딩)
    반환: optim_tensor (길이 S), TP_entropy_encoding_scalable에서
         torch.argsort(optim_tensor, descending=True) 했을 때
         정확히 order_idx가 나오도록 하는 점수 벡터
    """
    S = order_idx.numel()
    scores = torch.empty(S, device=order_idx.device, dtype=torch.float32)
    # 큰 점수 -> 먼저 인코딩되도록 부여 (내림차순 정렬 기준)
    scores[order_idx] = torch.arange(S - 1, -1, -1, device=order_idx.device, dtype=torch.float32)
    return scores

def get_transmission_tensor_optimal_channel(
    i, maxL,
    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
    l_ele, Nary_tensor,
    means_hat, pmf_center_list,
    net, clf, _transform, x_orig, pads, device,
    im_shape,  # (B=1, C, H, W)
    label,     # <-- 추가: GT label
    y_hat_base=None,
):
    """
    채널 단위 그룹:
      - 그룹 전송 시 CE 감소(ΔCE) 측정 (GT label 사용)
      - 그룹 비트(ΔR = ∑ H(p_k))로 나눔 → ΔCE/ΔR
      - 채널 간: ΔCE/ΔR 내림차순
      - 채널 내부: 기존 -(ΔD/ΔR) 내림차순
    반환:
      optim_tensor_custom (점수벡터), pmfs_norm
    """
    B, C, H, W = im_shape

    # A) 기본 -(ΔD/ΔR) 점수 + pmfs_norm
    optim_tensor_opt, pmfs_norm = get_transmission_tensor(
        i, maxL, pmfs_list, xpmfs_list, x2pmfs_list
    )
    total_symbols = optim_tensor_opt.numel()

    # B) concat 기준 플랫 인덱스들 (j=0..i 순서로 이어붙임)
    concat_idx = torch.cat([
        torch.nonzero((l_ele.reshape(-1) == (maxL - j)), as_tuple=False).squeeze(1)
        for j in range(i + 1)
    ], dim=0).to(device)  # (total_symbols,)

    # C) 채널별 상대 인덱스 묶기
    HW = H * W
    c_all = (concat_idx // HW) % C
    ch_lists = [[] for _ in range(C)]
    for rel in range(total_symbols):
        ch_lists[int(c_all[rel].item())].append(rel)
    ch_lists = [
        torch.tensor(v, device=device, dtype=torch.long) if len(v) > 0
        else torch.empty(0, device=device, dtype=torch.long)
        for v in ch_lists
    ]

    # D) dp-1까지 반영된 y_hat_base
    if y_hat_base is None:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                         xpmfs_list, pmfs_list, pmf_center_list))
        y_hat_base = means_hat.clone()
        for j in range(i):
            y_hat_base[l_ele == (l_ele.max() - j)] += recon[j]

    # E) 기준 CE (GT label 사용)
    target = _ensure_label_tensor(label, device)
    with torch.no_grad():
        x_base = net.g_s(y_hat_base, index_channel=0).clamp_(0, 1)
        x_base = crop_with_pads(x_base, pads)
        base_in = _transform(tensor_to_pil(x_base)).unsqueeze(0).to(device)
        out_base = clf(base_in)
        ce_base = F.cross_entropy(out_base, target)

    # F) 이번 plane 평균 shift (근사)
    recon_i = (xpmfs_list[i].sum(-1) / pmfs_list[i].sum(-1)) - pmf_center_list[i]
    shift_val = float(recon_i.mean().item()) if recon_i.ndim > 0 else float(recon_i.item())

    # G) 각 심볼 엔트로피(=비트) H(p)
    H_concat = _symbol_entropies_from_pmfs_norm(pmfs_norm)  # (total_symbols,)

    # H) 채널별 ΔCE/ΔR
    ch_score = torch.full((C,), -1e9, device=device)
    for ch in range(C):
        rels = ch_lists[ch]
        if rels.numel() == 0:
            continue

        idx_flat = concat_idx[rels]
        y_cand = y_hat_base.clone().reshape(-1)
        y_cand[idx_flat] = y_cand[idx_flat] + shift_val
        y_cand = y_cand.reshape(means_hat.shape)

        with torch.no_grad():
            x_cand = net.g_s(y_cand, index_channel=0).clamp_(0, 1)
            x_cand = crop_with_pads(x_cand, pads)
            cand_in = _transform(tensor_to_pil(x_cand)).unsqueeze(0).to(device)
            out_c = clf(cand_in)
            ce_c = F.cross_entropy(out_c, target)

        d_ce = (ce_base - ce_c).item()
        d_r  = float(H_concat[rels].sum().item()) + 1e-12
        ch_score[ch] = d_ce / d_r

    # I) 채널 순서: ΔCE/ΔR 내림차순
    ch_order = torch.argsort(ch_score, descending=True)

    # J) 최종 concat 상대 인덱스 순서: 채널 순서 → 내부는 -(ΔD/ΔR) 내림차순
    final_rel = []
    for ch in ch_order.tolist():
        rels = ch_lists[ch]
        if rels.numel() == 0:
            continue
        vals = optim_tensor_opt[rels]
        rel_sorted = rels[torch.argsort(vals, descending=True)]
        final_rel.append(rel_sorted)
    sort_idx = torch.cat(final_rel, dim=0) if len(final_rel) else torch.empty(0, device=device, dtype=torch.long)

    # K) 정렬 인덱스를 점수로 변환해 반환
    optim_tensor_custom = _scores_from_order(sort_idx)
    return optim_tensor_custom, pmfs_norm

def get_transmission_tensor_optimal_patch(
    i, maxL,
    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
    l_ele, Nary_tensor,
    means_hat, pmf_center_list,
    net, clf, _transform, x_orig, pads, device,
    im_shape,          # (B=1, C, H, W)
    label,             # <-- 추가: GT label
    y_hat_base=None,
    patch_h: int = 2, patch_w: int = 2,
):
    """
    2x2 패치 단위 그룹 최적화 (ΔCE/ΔR):
      - 그룹 간: ΔCE/ΔR 내림차순
      - 그룹 내부: -(ΔD/ΔR) 내림차순
    반환:
      optim_tensor_custom (점수벡터), pmfs_norm
    """
    B, C, H, W = im_shape
    assert H % patch_h == 0 and W % patch_w == 0, "H,W must be divisible by patch size"

    # A) 기본 -(ΔD/ΔR) 점수 + pmfs_norm
    optim_tensor_opt, pmfs_norm = get_transmission_tensor(
        i, maxL, pmfs_list, xpmfs_list, x2pmfs_list
    )
    total_symbols = optim_tensor_opt.numel()

    # B) concat 기준 플랫 인덱스들
    concat_idx = torch.cat([
        torch.nonzero((l_ele.reshape(-1) == (maxL - j)), as_tuple=False).squeeze(1)
        for j in range(i + 1)
    ], dim=0).to(device)  # (total_symbols,)

    # C) (c,h,w) 얻기
    HW = H * W
    c_all = (concat_idx // HW) % C
    h_all = (concat_idx % HW) // W
    w_all = (concat_idx % HW) % W

    # D) 패치 그리드
    PH, PW = H // patch_h, W // patch_w
    patch_lists = [ [] for _ in range(PH * PW) ]
    for rel in range(total_symbols):
        ph = int(h_all[rel].item()) // patch_h
        pw = int(w_all[rel].item()) // patch_w
        pid = ph * PW + pw
        patch_lists[pid].append(rel)
    patch_lists = [
        torch.tensor(v, device=device, dtype=torch.long) if len(v) > 0
        else torch.empty(0, device=device, dtype=torch.long)
        for v in patch_lists
    ]

    # E) dp-1까지 반영된 y_hat_base
    if y_hat_base is None:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                         xpmfs_list, pmfs_list, pmf_center_list))
        y_hat_base = means_hat.clone()
        for j in range(i):
            y_hat_base[l_ele == (l_ele.max() - j)] += recon[j]

    # F) 기준 CE (GT label 사용)
    target = _ensure_label_tensor(label, device)
    with torch.no_grad():
        x_base = net.g_s(y_hat_base, index_channel=0).clamp_(0, 1)
        x_base = crop_with_pads(x_base, pads)
        base_in = _transform(tensor_to_pil(x_base)).unsqueeze(0).to(device)
        out_base = clf(base_in)
        ce_base = F.cross_entropy(out_base, target)

    # G) 이번 plane 평균 shift (근사)
    recon_i = (xpmfs_list[i].sum(-1) / pmfs_list[i].sum(-1)) - pmf_center_list[i]
    shift_val = float(recon_i.mean().item()) if recon_i.ndim > 0 else float(recon_i.item())

    # H) 각 심볼 엔트로피(=비트)
    H_concat = _symbol_entropies_from_pmfs_norm(pmfs_norm)

    # I) 패치별 ΔCE/ΔR
    patch_score = torch.full((PH * PW,), -1e9, device=device)
    for pid in range(PH * PW):
        rels = patch_lists[pid]
        if rels.numel() == 0:
            continue

        idx_flat = concat_idx[rels]
        y_cand = y_hat_base.clone().reshape(-1)
        y_cand[idx_flat] = y_cand[idx_flat] + shift_val
        y_cand = y_cand.reshape(means_hat.shape)

        with torch.no_grad():
            x_cand = net.g_s(y_cand, index_channel=0).clamp_(0, 1)
            x_cand = crop_with_pads(x_cand, pads)
            cand_in = _transform(tensor_to_pil(x_cand)).unsqueeze(0).to(device)
            out_c = clf(cand_in)
            ce_c = F.cross_entropy(out_c, target)

        d_ce = (ce_base - ce_c).item()
        d_r  = float(H_concat[rels].sum().item()) + 1e-12
        patch_score[pid] = d_ce / d_r

    # J) 패치 순서: ΔCE/ΔR 내림차순
    patch_order = torch.argsort(patch_score, descending=True)

    # K) 최종 concat 상대 인덱스 순서: 패치 순서 → 내부는 -(ΔD/ΔR) 내림차순
    final_rel = []
    for pid in patch_order.tolist():
        rels = patch_lists[pid]
        if rels.numel() == 0:
            continue
        vals = optim_tensor_opt[rels]
        rel_sorted = rels[torch.argsort(vals, descending=True)]
        final_rel.append(rel_sorted)
    sort_idx = torch.cat(final_rel, dim=0) if len(final_rel) else torch.empty(0, device=device, dtype=torch.long)

    # L) 정렬 인덱스를 점수 벡터로 변환해 반환
    optim_tensor_custom = _scores_from_order(sort_idx)
    return optim_tensor_custom, pmfs_norm

def get_transmission_tensor_random(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list, seed):
    p_len = mode ** (maxL - 1 - i)

    pmfs_list_l = pmfs_list[:i + 1]
    xpmfs_list_l = xpmfs_list[:i + 1]
    x2pmfs_list_l = x2pmfs_list[:i + 1]

    pmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in pmfs_list_l]
    xpmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in xpmfs_list_l]
    pmfs_norm = [p / p.sum(-1, keepdim=True) for p in pmfs_cond_list_l]

    x2pmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in x2pmfs_list_l]
    m_old = [xp.sum(-1) / p.sum(-1) for xp, p in zip(xpmfs_list_l, pmfs_list_l)]
    D_old = [(x2.sum(-1) - (m**2) * p.sum(-1)) / p.sum(-1)
             for x2, p, m in zip(x2pmfs_list_l, pmfs_list_l, m_old)]
    m_new = [xp / p for xp, p in zip(xpmfs_cond_list_l, pmfs_cond_list_l)]
    D_new = [((x2 - (m**2) * p) / fullp.sum(-1, keepdim=True)).sum(-1)
             for x2, p, m, fullp in zip(x2pmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l)]
    delta_D = [(old - new).clamp(max=0) for old, new in zip(D_old, D_new)]
    delta_R = [(-p * torch.log2(p)).sum(-1).clamp(min=0) for p in pmfs_norm]

    # 길이 S
    S = torch.cat([-(D / R) for D, R in zip(delta_D, delta_R)]).numel()
    g = torch.Generator(device=pmfs_list_l[0].device)
    g.manual_seed(seed)
    order_rand = torch.randperm(S, generator=g, device=pmfs_list_l[0].device)
    optim_tensor_rand = _scores_from_order(order_rand)

    return optim_tensor_rand, pmfs_norm


def get_transmission_tensor_reverse(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list):
    p_len = mode ** (maxL - 1 - i)

    pmfs_list_l = pmfs_list[:i + 1]
    xpmfs_list_l = xpmfs_list[:i + 1]
    x2pmfs_list_l = x2pmfs_list[:i + 1]

    pmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in pmfs_list_l]
    xpmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in xpmfs_list_l]
    pmfs_norm = [p / p.sum(-1, keepdim=True) for p in pmfs_cond_list_l]

    x2pmfs_cond_list_l = [x.view(x.size(0), mode, p_len).sum(-1) for x in x2pmfs_list_l]
    m_old = [xp.sum(-1) / p.sum(-1) for xp, p in zip(xpmfs_list_l, pmfs_list_l)]
    D_old = [(x2.sum(-1) - (m**2) * p.sum(-1)) / p.sum(-1)
             for x2, p, m in zip(x2pmfs_list_l, pmfs_list_l, m_old)]
    m_new = [xp / p for xp, p in zip(xpmfs_cond_list_l, pmfs_cond_list_l)]
    D_new = [((x2 - (m**2) * p) / fullp.sum(-1, keepdim=True)).sum(-1)
             for x2, p, m, fullp in zip(x2pmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l)]
    delta_D = [(old - new).clamp(max=0) for old, new in zip(D_old, D_new)]
    delta_R = [(-p * torch.log2(p)).sum(-1).clamp(min=0) for p in pmfs_norm]

    # 원래 optimal 점수 (내림차순 정렬 시 optimal 순서)
    optim_tensor_opt = torch.cat([-(D / R) for D, R in zip(delta_D, delta_R)]).clamp(min=0)

    # 그 순서를 뒤집어 원하는 order를 만들고 → 점수 변환
    order_opt = torch.argsort(optim_tensor_opt, descending=True)
    order_rev = torch.flip(order_opt, dims=[0])
    optim_tensor_rev = _scores_from_order(order_rev)

    return optim_tensor_rev, pmfs_norm

def get_transmission_tensor_sigma(i, maxL, pmfs_list, scales_hat, l_ele):
    MODE = 3
    p_len = MODE ** (maxL - 1 - i)

    sigma_values = []
    for j in range(i + 1):
        sigmas = scales_hat.reshape(-1)[l_ele.reshape(-1) == maxL - j]
        sigma_values.append(sigmas)
    sigma_cat = torch.cat(sigma_values)  # 길이 S

    # sigma 큰 순서 → order → 점수로 변환
    order_sigma = torch.argsort(sigma_cat, descending=True)
    optim_tensor_sigma = _scores_from_order(order_sigma)

    pmfs_list_l = pmfs_list[:i + 1]
    pmfs_cond_list_l = [x.view(x.size(0), MODE, p_len).sum(-1) for x in pmfs_list_l]
    pmfs_norm = [p / p.sum(-1, keepdim=True) for p in pmfs_cond_list_l]
    return optim_tensor_sigma, pmfs_norm

def reconstruct_y_hat_from_digits(Nary_tensor, l_ele, means_hat, mode: int, maxL: int):
    device = Nary_tensor.device
    flat_means = means_hat.reshape(-1)
    flat_l = l_ele.reshape(-1)
    y_hat_flat = flat_means.clone()

    unique_L = torch.unique(flat_l)
    for L in unique_L.tolist():
        idx = (flat_l == L).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        digits = Nary_tensor[idx, maxL - L: maxL].to(torch.int64)  # (n_L, L)
        weights = (mode ** torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64))  # (L,)
        symbols = (digits * weights).sum(dim=1)  # (n_L,)
        centers = (mode ** L) // 2
        offsets = symbols - centers
        y_hat_flat[idx] = flat_means[idx] + offsets.to(flat_means.dtype)

    return y_hat_flat.reshape(means_hat.shape)

def TP_entropy_decoding_light(i, device, maxL, l_ele, Nary_tensor,
                              pmfs_list, idx_ts_list,
                              pmfs_norm,
                              decoder, means_hat, mode, is_recon):
    """
    xpmfs/x2pmfs 없이 pmfs만으로 디코딩.
    마지막에 is_recon=True면, 자릿수 합산으로 y_hat 직접 복원.
    """
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat([_pmf_to_cdf_tensor(p, tm) for p, tm in zip(pmfs_norm, tail_mass)], dim=0).tolist()

    symbols_num = (l_ele.reshape(-1) >= maxL - i).sum().item()
    indexes_list = list(range(symbols_num))
    cdf_lengths = [mode + 2 for _ in range(symbols_num)]
    offsets = [-(mode // 2) for _ in range(symbols_num)]

    rv = decoder.decode_stream(indexes_list, cond_cdf, cdf_lengths, offsets)
    rv = (torch.Tensor(rv) - torch.Tensor(offsets)).int().to(device)

    # 스트림을 레벨별로 분배
    tmp_idx = 0
    for j in range(i + 1):
        num = len(pmfs_list[j])
        if j == 0:
            tmp_idx += num
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:tmp_idx + num]
            tmp_idx += num

    # 다음 단계 준비 (pmf 서브인터벌 축소)
    select_sub_interval_light(i, device, maxL, l_ele, Nary_tensor, pmfs_list, idx_ts_list, mode=mode)

    if is_recon:
        # 자릿수 합산으로 직접 복원
        y_hat = reconstruct_y_hat_from_digits(Nary_tensor, l_ele.reshape(-1), means_hat, mode, maxL)
    else:
        y_hat = -1

    return y_hat

def TPED_light(i, device, maxL, l_ele, Nary_tensor,
               pmfs_list, idx_ts_list,
               optim_tensor,
               point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
               decoder, decoded_rvs,
               means_hat, mode, is_recon):
    """
    기존 TPED와 동일한 흐름이지만, pmfs만 유지하며 자릿수 합산 복원 사용.
    """
    # 이번 청크 길이
    indexes_list = list(range(sl))
    rv = decoder.decode_stream(indexes_list,
                               cond_cdf[point * sl:(point + 1) * sl],
                               cdf_lengths[point * sl:(point + 1) * sl],
                               offsets[point * sl:(point + 1) * sl])
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:(point + 1) * sl])).int().to(device)
    decoded_rvs.append(rv.clone())

    # 아직 남은 심볼들은 -1로 채워서 길이를 total_symbols로 맞춘다
    pre_cat = torch.cat(decoded_rvs)
    post_cat = torch.zeros([total_symbols - (point + 1) * sl], device=device) - 1
    rv_full = torch.cat([pre_cat, post_cat])

    # 엔코더에서의 정렬(optim_tensor 내림차순)을 역정렬하여 원래 순서로 배치
    Nary_tensor_tmp = rv_full[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()

    # 레벨별로 잘라 넣기
    tmp_idx = 0
    for j in range(i + 1):
        num = len(pmfs_list[j])
        if j == 0:
            tmp_idx += num
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:tmp_idx + num]
            tmp_idx += num

    # 이미 디코드된(= -1 아님) 위치만 서브인터벌 축소 반영
    for j in range(i + 1):
        row_mask = (l_ele.reshape(-1) == maxL - j)
        col_vals = Nary_tensor[row_mask, i]
        valid_mask = (col_vals != -1)
        if valid_mask.any():
            vals = col_vals[valid_mask].view(-1, 1)
            p_len = mode ** (maxL - 1 - i)
            tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(vals.size(0), 1).int()
            idx_ts_list[j][valid_mask] *= (torch.div((tmp_ % (p_len * mode)), p_len, rounding_mode="floor") == vals)
            pmfs_list[j][valid_mask] *= idx_ts_list[j][valid_mask]

    Nary_tensor[Nary_tensor < 0] = 0

    if is_recon:
        y_hat = reconstruct_y_hat_from_digits(Nary_tensor, l_ele.reshape(-1), means_hat, mode, maxL)
    else:
        y_hat = -1

    return y_hat

def TPED_last_point_light(i, device, maxL, l_ele, Nary_tensor,
                          pmfs_list, idx_ts_list,
                          optim_tensor,
                          point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                          decoder, decoded_rvs,
                          means_hat, mode, is_recon):
    p_len = mode ** (maxL - 1 - i)

    symbols_num_part = total_symbols - point * sl
    indexes_list = list(range(symbols_num_part))
    rv = decoder.decode_stream(indexes_list,
                               cond_cdf[point * sl:],
                               cdf_lengths[point * sl:],
                               offsets[point * sl:])
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:])).int().to(device)
    decoded_rvs.append(rv.clone())
    rv_full = torch.cat(decoded_rvs)

    # 역정렬로 원위치
    Nary_tensor_tmp = rv_full[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()

    # 레벨별 분배
    tmp_idx = 0
    for j in range(i + 1):
        num = len(pmfs_list[j])
        if j == 0:
            tmp_idx += num
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:tmp_idx + num]
            tmp_idx += num

    # 서브인터벌 축소
    for j in range(i + 1):
        row_mask = (l_ele.reshape(-1) == maxL - j)
        vals = Nary_tensor[row_mask, i].view(-1, 1)
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(vals.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (mode * p_len)), p_len, rounding_mode="floor") == vals)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        num_pmf = pmfs_list[j].size(0)
        size_pmf = pmfs_list[j].size(1) // mode
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, size_pmf)

    if is_recon:
        y_hat = reconstruct_y_hat_from_digits(Nary_tensor, l_ele.reshape(-1), means_hat, mode, maxL)
    else:
        y_hat = -1

    return y_hat

def get_ans(type):
    if type == "enc":
        return BufferedRansEncoder()
    elif type == "dec":
        return RansDecoder()
    else:
        raise ValueError(f"type must be 'enc' or 'dec'")

def _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
    cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        _cdf = _pmf_to_quantized_cdf(prob.tolist(), 16)
        _cdf = torch.IntTensor(_cdf)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf


def _pmf_to_cdf_tensor(pmf, tail_mass):
    _cdf = torch.cat([pmf, tail_mass], dim=1).clamp_(min=2e-05)
    _cdf = torch.round((_cdf / _cdf.sum(dim=1, keepdim=True)).cumsum(dim=1) * (2 ** 16))
    _cdf = torch.cat([torch.zeros_like(tail_mass), _cdf], dim=1).int()
    return _cdf


def _standardized_cumulative(inputs):
    half = float(0.5)
    const = float(-(2 ** -0.5))
    # Using the complementary error function maximizes numerical precision.
    return half * torch.erfc(const * inputs)


def _pnum_part(i, max_L):
    if i == max_L - 0:
        pnum_part = 24 / 24
    elif i == max_L - 1:
        pnum_part = 24 / 24
    elif i == max_L - 2:
        pnum_part = 24 / 24
    elif i == max_L - 3:
        pnum_part = 16 / 24
    elif i == max_L - 4:
        pnum_part = 8 / 24
    elif i == max_L - 5:
        pnum_part = 8 / 24
    elif i == max_L - 6:
        pnum_part = 3 / 48
    else:
        pnum_part = 1 / 48
    return pnum_part


def get_empty_Nary_tensor(scales_hat):
    device = scales_hat.device
    tail = scales_hat * multiplier * 2
    l_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
    l_ele = torch.clamp(l_ele, 1, l_ele.max().item())
    maxL = l_ele.max().item()

    if torch.sum(l_ele == l_ele.max()) < 2:
        maxL = l_ele.max().item() - 1
        l_ele = torch.clamp(l_ele, 1, maxL)

    Nary_tensor = torch.zeros(list(scales_hat.shape) + [maxL]).int().to(device)
    return device, maxL, l_ele, Nary_tensor


def get_Nary_tensor(y, means_hat, scales_hat):
    device, maxL, l_ele, Nary_tensor = get_empty_Nary_tensor(scales_hat)

    symbol_tensor = torch.round(y - means_hat).int() + torch.div(mode ** l_ele, 2, rounding_mode="floor")
    symbol_tensor = torch.clamp(symbol_tensor, min=torch.zeros(y.shape).int().to(device), max=3 ** l_ele - 1)

    for i in range(1, maxL + 1):
        Nary_tensor[:, :, :, :, i - 1] = torch.div(symbol_tensor, (mode ** (maxL - i)), rounding_mode="floor")
        symbol_tensor = symbol_tensor % (mode ** (maxL - i))

    Nary_tensor = Nary_tensor.view(-1, maxL)
    del symbol_tensor
    torch.cuda.empty_cache()

    return device, maxL, l_ele, Nary_tensor


def make_pmf_table(scales_hat, device, maxL, l_ele):
    pmfs_list = []
    xpmfs_list = []
    x2pmfs_list = []
    idx_ts_list = []

    for i in range(1, maxL + 1):
        pmf_length = mode ** i
        pmf_center = pmf_length // 2
        samples = torch.abs(torch.arange(pmf_length, device=device).repeat((l_ele == i).sum(), 1) - pmf_center)
        upper = _standardized_cumulative((0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        lower = _standardized_cumulative((-0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        pmfs_ = upper - lower
        pmfs_ = (pmfs_ + 1e-10) / (pmfs_ + 1e-10).sum(dim=-1).unsqueeze(-1)
        pmfs_list.insert(0, pmfs_.clone())
        del upper, lower, samples
        torch.cuda.empty_cache()
        idx_tmp = torch.arange(mode ** i, device=device).repeat(pmfs_.size(0), 1)
        xpmfs_ = pmfs_ * idx_tmp
        xpmfs_list.insert(0, xpmfs_.clone())
        x2pmfs_ = pmfs_ * torch.pow(idx_tmp, 2)
        x2pmfs_list.insert(0, x2pmfs_.clone())
        idx_ts_list.insert(0, torch.ones_like(pmfs_list[0], device=device))
        del idx_tmp, pmfs_, xpmfs_, x2pmfs_
        torch.cuda.empty_cache()

    return pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list

def make_pmf_table_light(scales_hat, device, maxL, l_ele, mode=3):
    pmfs_list = []
    idx_ts_list = []
    for i in range(1, maxL + 1):
        pmf_length = mode ** i
        pmf_center = pmf_length // 2
        samples = torch.abs(torch.arange(pmf_length, device=device).repeat((l_ele == i).sum(), 1) - pmf_center)
        upper = _standardized_cumulative((0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        lower = _standardized_cumulative((-0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        pmfs_ = upper - lower
        pmfs_ = (pmfs_ + 1e-10) / (pmfs_ + 1e-10).sum(dim=-1).unsqueeze(-1)
        pmfs_list.insert(0, pmfs_.clone())
        idx_ts_list.insert(0, torch.ones_like(pmfs_list[0], device=device))
        del upper, lower, samples, pmfs_
        torch.cuda.empty_cache()
    return pmfs_list, idx_ts_list

def select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list):
    p_len = mode ** (maxL - 1 - i)

    for j in range(i + 1):
        num_pmf = pmfs_list[j].size(0)
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i:i + 1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (p_len * mode)), p_len, rounding_mode="floor") == Nary_part)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, p_len)
        xpmfs_list[j] = xpmfs_list[j][nz_idx].view(num_pmf, p_len)
        x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(num_pmf, p_len)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, p_len)

def select_sub_interval_light(i, device, maxL, l_ele, Nary_tensor, pmfs_list, idx_ts_list, mode=3):
    p_len = mode ** (maxL - 1 - i)
    for j in range(i + 1):
        num_pmf = pmfs_list[j].size(0)
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i:i + 1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (p_len * mode)), p_len, rounding_mode="floor") == Nary_part)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, p_len)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, p_len)

def get_transmission_tensor(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list):
    p_len = mode ** (maxL - 1 - i)

    pmfs_list_l = pmfs_list[:i + 1]
    xpmfs_list_l = xpmfs_list[:i + 1]
    x2pmfs_list_l = x2pmfs_list[:i + 1]
    m_old = list(map(lambda x, y: x.sum(dim=-1) / y.sum(dim=-1), xpmfs_list_l, pmfs_list_l))
    D_old = list(map(lambda x2p, p, m: (x2p.sum(-1) - (m ** 2) * p.sum(-1)) / p.sum(-1), x2pmfs_list_l, pmfs_list_l, m_old))

    pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), mode, p_len).sum(-1), pmfs_list_l))
    xpmfs_cond_list_l = list(map(lambda xp: xp.view(xp.size(0), mode, p_len).sum(-1), xpmfs_list_l))
    x2pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), mode, p_len).sum(-1), x2pmfs_list_l))

    pmfs_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))

    m_new = list(map(lambda xp, p: xp / p, xpmfs_cond_list_l, pmfs_cond_list_l))
    D_new = list(map(lambda x2p, p, m, fullp: ((x2p - (m ** 2) * p) / fullp.sum(-1).view(-1, 1)).sum(-1),
                     x2pmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l))
    delta_D = list(map(lambda old, new: (new - old).clamp_(max=0), D_old, D_new))
    delta_R = list(map(lambda p: (-p * torch.log2(p)).sum(-1), pmfs_norm))
    delta_R = list(map(lambda h: h * (h >= 0), delta_R))

    optim_tensor = torch.cat(list(map(lambda D, R: -(D / R), delta_D, delta_R))).clamp_(min=0)

    return optim_tensor, pmfs_norm


def TP_entropy_encoding(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        pmfs_norm,
                        encoder, y_strings):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0).tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)]).tolist()
    indexes_list = list(range(len(total_symbols_list)))
    cdf_lengths = [mode + 2 for _ in range(len(total_symbols_list))]

    offsets = [-(mode // 2) for _ in range(len(total_symbols_list))]
    encoder.encode_with_indexes(
        total_symbols_list, indexes_list, cond_cdf, cdf_lengths, offsets
    )
    del pmfs_norm, tail_mass, cond_cdf
    torch.cuda.empty_cache()
    y_strings[i].append(encoder.flush())

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)

def TP_entropy_encoding_light(i, device, maxL, l_ele, Nary_tensor,
                              pmfs_list, idx_ts_list,
                              pmfs_norm,
                              encoder, y_strings, mode=3):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0).tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)]).tolist()
    indexes_list = list(range(len(total_symbols_list)))
    cdf_lengths = [mode + 2 for _ in range(len(total_symbols_list))]
    offsets = [-(mode // 2) for _ in range(len(total_symbols_list))]

    encoder.encode_with_indexes(total_symbols_list, indexes_list, cond_cdf, cdf_lengths, offsets)
    y_strings[i].append(encoder.flush())

    select_sub_interval_light(i, device, maxL, l_ele, Nary_tensor, pmfs_list, idx_ts_list, mode=mode)

def TP_entropy_encoding_scalable(i, device, maxL, l_ele, Nary_tensor,
                                 pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                                 pmfs_norm, optim_tensor,
                                 encoder, y_strings):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0)
    cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)])
    total_symbols_list = total_symbols_list[torch.argsort(optim_tensor, descending=True)].tolist()
    total_symbols = len(total_symbols_list)
    cdf_lengths = [mode + 2 for _ in range(total_symbols)]
    offsets = [-(mode // 2) for _ in range(total_symbols)]

    torch.cuda.empty_cache()

    # sl = total_symbols // pnum_btw_trit
    pnum_part = _pnum_part(i, maxL)
    points_num = math.ceil(pnum_btw_trit * pnum_part)
    
    # Ensure points_num doesn't exceed total_symbols to avoid sl=0
    points_num = min(points_num, total_symbols) if total_symbols > 0 else 1

    sl = total_symbols // points_num if points_num > 0 else 0

    for point in range(points_num):
        if point == points_num - 1:
            symbols_list = total_symbols_list[point * sl:]
            indexes_list = list(range(len(symbols_list)))
            encoder.encode_with_indexes(
                symbols_list,
                indexes_list,
                cond_cdf[point * sl:],
                cdf_lengths[point * sl:],
                offsets[point * sl:]
            )
            y_strings[i].append(encoder.flush())
            break

        symbols_list = total_symbols_list[point * sl:(point + 1) * sl]
        indexes_list = list(range(len(symbols_list)))
        encoder.encode_with_indexes(
            symbols_list,
            indexes_list,
            cond_cdf[point * sl:(point + 1) * sl],
            cdf_lengths[point * sl:(point + 1) * sl],
            offsets[point * sl:(point + 1) * sl]
        )
        y_strings[i].append(encoder.flush())

        encoder = BufferedRansEncoder()

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)

def TP_entropy_encoding_scalable_light(i, device, maxL, l_ele, Nary_tensor,
                                       pmfs_list, idx_ts_list,
                                       pmfs_norm, optim_tensor,
                                       encoder, y_strings, mode=3):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0)
    cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)])
    total_symbols_list = total_symbols_list[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols = len(total_symbols_list)
    cdf_lengths = [mode + 2 for _ in range(total_symbols)]
    offsets = [-(mode // 2) for _ in range(total_symbols)]

    pnum_part_val = _pnum_part(i, maxL)
    points_num = max(1, math.ceil(pnum_btw_trit * pnum_part_val))
    
    # Ensure points_num doesn't exceed total_symbols to avoid sl=0
    points_num = min(points_num, total_symbols) if total_symbols > 0 else 1
    
    sl = total_symbols // points_num if points_num > 0 else 0

    for point in range(points_num):
        if point == points_num - 1:
            symbols_list = total_symbols_list[point * sl:]
            indexes_list = list(range(len(symbols_list)))
            encoder.encode_with_indexes(
                symbols_list, indexes_list,
                cond_cdf[point * sl:], cdf_lengths[point * sl:], offsets[point * sl:]
            )
            y_strings[i].append(encoder.flush())
            break

        symbols_list = total_symbols_list[point * sl:(point + 1) * sl]
        indexes_list = list(range(len(symbols_list)))
        encoder.encode_with_indexes(
            symbols_list, indexes_list,
            cond_cdf[point * sl:(point + 1) * sl],
            cdf_lengths[point * sl:(point + 1) * sl],
            offsets[point * sl:(point + 1) * sl]
        )
        y_strings[i].append(encoder.flush())
        encoder = BufferedRansEncoder()

    select_sub_interval_light(i, device, maxL, l_ele, Nary_tensor, pmfs_list, idx_ts_list, mode=mode)

def TP_entropy_decoding(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        pmfs_norm,
                        decoder, means_hat, pmf_center_list, is_recon):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0).tolist()

    symbols_num = (l_ele.reshape(-1) >= maxL - i).sum().item()
    indexes_list = list(range(symbols_num))
    cdf_lengths = [mode + 2 for _ in range(symbols_num)]
    offsets = [-(mode // 2) for _ in range(symbols_num)]
    rv = decoder.decode_stream(
        indexes_list, cond_cdf, cdf_lengths, offsets
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets)).int().to(device)
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat


def prepare_TPED_scalable(i, device, maxL, l_ele,
                           pmfs_norm, optim_tensor):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0)
    cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols = (l_ele.reshape(-1) >= maxL - i).sum().item()
    cdf_lengths = [mode + 2 for _ in range(total_symbols)]
    offsets = [-(mode // 2) for _ in range(total_symbols)]
    del tail_mass

    torch.cuda.empty_cache()

    pnum_part = _pnum_part(i, maxL)
    points_num = math.ceil(pnum_btw_trit * pnum_part)

    sl = total_symbols // points_num

    return cond_cdf, total_symbols, cdf_lengths, offsets, sl, points_num


def TPED_last_point(i, device, maxL, l_ele, Nary_tensor,
                     pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                     optim_tensor,
                     point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                     decoder, decoded_rvs,
                     means_hat, pmf_center_list, is_recon):
    p_len = mode ** (maxL - 1 - i)

    symbols_num_part = total_symbols - point * sl
    indexes_list = list(range(symbols_num_part))
    rv = decoder.decode_stream(
        indexes_list,
        cond_cdf[point * sl:],
        cdf_lengths[point * sl:],
        offsets[point * sl:]
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:])).int().to(device)
    decoded_rvs.append(rv.clone())
    rv = torch.cat(decoded_rvs)
    Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    for j in range(i + 1):
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i:i + 1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (mode * p_len)), p_len, rounding_mode="floor") == Nary_part)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        num_pmf = pmfs_list[j].size(0)
        size_pmf = pmfs_list[j].size(1) // mode
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        xpmfs_list[j] = xpmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, size_pmf)

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                         xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat


def TPED(i, device, maxL, l_ele, Nary_tensor,
          pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
          optim_tensor,
          point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
          decoder, decoded_rvs,
          means_hat, pmf_center_list, is_recon):
    p_len = mode ** (maxL - 1 - i)

    indexes_list = list(range(sl))
    rv = decoder.decode_stream(
        indexes_list,
        cond_cdf[point * sl:(point + 1) * sl],
        cdf_lengths[point * sl:(point + 1) * sl],
        offsets[point * sl:(point + 1) * sl]
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:(point + 1) * sl])).int().to(device)
    decoded_rvs.append(rv.clone())

    pre_cat = torch.cat(decoded_rvs)
    post_cat = torch.zeros([total_symbols - (point + 1) * sl]).to(device) - 1
    rv = torch.cat([pre_cat, post_cat])

    Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = \
                Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    for j in range(i + 1):
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= (torch.div((tmp_ % (p_len * 3)), p_len, rounding_mode="floor") == Nary_part.view(-1, 1))
        pmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        xpmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        x2pmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
    Nary_tensor[Nary_tensor < 0] = 0

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat
