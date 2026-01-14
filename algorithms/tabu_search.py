from collections import Counter, deque
from itertools import combinations

def tabu_search(transactions, max_iter=15, tabu_size=5, k=5, threshold=0.5):
    """
    Tabu Search untuk memilih k item fokus dari transaksi dan mem-filter
    transaksi sebelum FP-Growth.

    Args:
        transactions (List[List[str]]): daftar transaksi (list item per struk)
        max_iter (int): jumlah iterasi maksimal
        tabu_size (int): ukuran tabu list (tenure)
        k (int): jumlah item fokus yang dipilih
        threshold (float): minimum support relatif (0..1) untuk kandidat item

    Returns:
        dict: {
          'selected_items': List[str],
          'filtered_transactions': List[List[str]],
          'score': float
        }
    """
    n = len(transactions)
    if n == 0:
        return {'selected_items': [], 'filtered_transactions': [], 'score': 0.0}

    # Gunakan set per transaksi untuk menghitung presence dengan benar
    trx_sets = [set(t) for t in transactions]

    # Support single items (presence-based)
    item_counts = Counter()
    for s in trx_sets:
        item_counts.update(s)
    item_support = {it: item_counts[it] / n for it in item_counts}

    # Support pasangan (co-occurrence presence-based)
    pair_counts = Counter()
    for s in trx_sets:
        for a, b in combinations(sorted(s), 2):
            pair_counts[(a, b)] += 1
    pair_support = {p: pair_counts[p] / n for p in pair_counts}

    # Kandidat awal berdasar threshold; fallback jika kosong
    universe = [it for it, sup in item_support.items() if sup >= threshold]
    if not universe:
        universe = [it for it, _ in item_counts.most_common()]

    # Inisialisasi solusi dengan top-k by support
    init = [it for it, _ in sorted(((it, item_support[it]) for it in universe),
                                   key=lambda x: (-x[1], x[0]))][:k]
    S = set(init)

    # Objective: gabungan support rata-rata dan ko-occurrence internal
    def score_set(Sset):
        if not Sset:
            return 0.0
        Slist = sorted(Sset)
        s1 = sum(item_support.get(it, 0.0) for it in Slist) / k
        if len(Slist) >= 2 and k >= 2:
            pairs = list(combinations(Slist, 2))
            denom_pairs = (k * (k - 1)) / 2
            s2 = sum(pair_support.get(tuple(sorted(p)), 0.0) for p in pairs)
            s2 = s2 / denom_pairs
        else:
            s2 = 0.0
        return 0.6 * s1 + 0.4 * s2

    best_S = set(S)
    best_score = score_set(best_S)

    tabu = deque(maxlen=tabu_size)

    for iter_count in range(max_iter):
        current_best_neighbor = None
        current_best_move = None
        current_best_score = -1.0

        # Generate neighbors: add/remove/swap
        # ADD
        if len(S) < k:
            for x in universe:
                if x in S:
                    continue
                move_token = ("add", x)
                cand = set(S)
                cand.add(x)
                sc = score_set(cand)
                if move_token in tabu and sc <= best_score:
                    continue  # tabu, no aspiration
                if sc > current_best_score:
                    current_best_score = sc
                    current_best_neighbor = cand
                    current_best_move = move_token

        # REMOVE
        for i in list(S):
            move_token = ("remove", i)
            cand = set(S)
            cand.remove(i)
            sc = score_set(cand)
            if move_token in tabu and sc <= best_score:
                continue
            if sc > current_best_score:
                current_best_score = sc
                current_best_neighbor = cand
                current_best_move = move_token

        # SWAP
        for i in list(S):
            for x in universe:
                if x in S:
                    continue
                move_token = ("swap", i, x)
                cand = set(S)
                cand.discard(i)
                cand.add(x)
                sc = score_set(cand)
                if move_token in tabu and sc <= best_score:
                    continue
                if sc > current_best_score:
                    current_best_score = sc
                    current_best_neighbor = cand
                    current_best_move = move_token

        # Jika tak ada neighbor valid, berhenti
        if current_best_neighbor is None:
            # print(f"[iter {iter_count}] no valid neighbor, stopping")
            break

        # Apply move
        S = current_best_neighbor
        tabu.append(current_best_move)

        # Update best (global)
        if current_best_score > best_score:
            best_S = set(S)
            best_score = current_best_score
            # print(f"[iter {iter_count}] new best score={best_score:.4f} move={current_best_move}")
        # else:
        #     print(f"[iter {iter_count}] accepted move={current_best_move} score={current_best_score:.4f}")

    # Filter transaksi ke item terpilih
    selected_items = sorted(best_S)
    filtered = []
    for t in transactions:
        # Omitted debug print for production API: print(f"[debug] t={t}")
        ft = [it for it in t if it in best_S]
        if ft:
            filtered.append(ft)

    return {
        'selected_items': selected_items,
        'filtered_transactions': filtered,
        'score': best_score
    }
