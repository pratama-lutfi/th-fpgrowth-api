import collections
import math
import itertools
import numpy as np
import pandas as pd
import warnings

class FPNode(object):
    """
    Node individual di dalam FP-Tree.
    """
    def __init__(self, item, count=0, parent=None):
        self.item = item          # Nama item (None untuk root)
        self.count = count        # Jumlah transaksi yang melewati node ini
        self.parent = parent      # Node induk
        self.children = collections.defaultdict(FPNode)  # Child nodes

        if parent is not None:
            parent.children[item] = self

    def get_item_path_from_root(self):
        """
        Ambil path item dari root (tidak termasuk node ini) dalam urutan root -> ... -> parent.
        
        Returns:
            list: item-item pada jalur, sudah diurutkan dari root
        """
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path

class FPTree(object):
    """
    Struktur data FP-Tree untuk menyimpan transaksi dalam bentuk pohon.
    Setiap node merepresentasikan item dengan count tertentu.
    """
    def __init__(self, rank=None):
        # Root node virtual (item=None)
        self.root = FPNode(None)
        # Dictionary menampung daftar node per item
        self.nodes = collections.defaultdict(list)
        # Item kondisi untuk conditional tree
        self.cond_items = []
        # Urutan prioritas item (support descending)
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        """
        Bangun conditional FP-Tree untuk item tertentu.
        
        Langkah-langkah:
        1. Kumpulkan semua path dari root ke node cond_item.
        2. Hitung count tiap item pada path-path tersebut.
        3. Filter item yang memenuhi minsup.
        4. Bangun conditional tree baru dengan item yang sudah difilter & diurutkan.
        
        Returns:
            FPTree baru yang dikondisikan pada cond_item
        """
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.get_item_path_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Item yang memenuhi minsup, diurutkan ascending (untuk kemudahan)
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Bangun conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )
            cond_tree.insert_ordered_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_ordered_itemset(self, itemset, count=1):
        """
        Masukkan itemset (urutan menurun support) ke dalam tree.
        
        Langkah-langkah:
        1. Naikkan count root.
        2. Untuk setiap item, jika child sudah ada naikkan count, jika tidak buat node baru.
        3. Simpan node pada dictionary nodes untuk akses cepat.
        """
        self.root.count += count

        if len(itemset) == 0:
            return

        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        # Sisanya buat cabang baru
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_single_path(self):
        """
        Cek apakah tree hanya berisi satu path lurus (untuk optimasi mining).
        
        Returns:
            bool: True bila tree hanya satu jalur dari root ke leaf
        """
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        """
        Cetak status proses mining untuk debugging.
        
        Args:
            count: jumlah itemset yang sudah ditemukan
            colnames: dict pemetaan index ke nama kolom
        """
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset dari tree yang dikondisikan pada item (%s)" % (count, cond_items),
            end="\n",
        )

def setup_fptree(df, min_support):
    """
    Membangun struktur FP-Tree dari DataFrame transaksi.
    """
    num_itemsets = len(df.index)

    # Deteksi format sparse
    is_sparse = False
    if hasattr(df, "sparse"):
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        itemsets = df.values

    # Buat masker untuk nilai yang hilang / tidak valid
    disabled = df.copy()
    disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
        (disabled == 0) | (disabled == 1), np.nan, 0
    )

    # Hitung support tiap item dengan mengabaikan baris disabled
    item_support = np.array(
        np.nansum(df.values, axis=0)
        / (float(num_itemsets) - np.nansum(disabled, axis=0))
    )
    item_support = item_support.reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]

    # Urutkan item berdasarkan support (descending)
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    # Hapus nilai nol pada sparse matrix untuk efisiensi
    if is_sparse:
        itemsets.eliminate_zeros()

    # Bangun FP-Tree
    tree = FPTree(rank)
    for i in range(num_itemsets):
        if is_sparse:
            nonnull = itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        # Filter item yang memenuhi min_support lalu urutkan sesuai rank
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_ordered_itemset(itemset)

    return tree, disabled, rank

def generate_itemsets_with_supports(generator, df, disabled, min_support, num_itemsets, colname_map):
    """
    Menghasilkan daftar itemset beserta nilai support-nya dari generator pola yang ditemukan.
    """
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        dec = disabled[:, iset]
        _dec = df.values[:, iset]

        if len(iset) == 1:
            supports.append((sup - np.nansum(dec)) / (num_itemsets - np.nansum(dec)))

        elif len(iset) > 1:
            denom = 0
            num = 0
            for i in range(dec.shape[0]):
                item_dsbl = list(dec[i, :])
                item_orig = list(_dec[i, :])

                if 1 in set(item_dsbl):
                    denom += 1

                    if (0 not in set(item_orig)) or (
                        all(np.isnan(x) for x in item_orig)
                    ):
                        num -= 1

            if num_itemsets - denom == 0:
                supports.append(0)
            else:
                supports.append((sup + num) / (num_itemsets - denom))

    res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
    res_df = res_df[res_df["support"] >= min_support]

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df

def check_valid_input(df, null_values=False):
    """
    Validasi keabsahan DataFrame input.
    """
    if df is None:
        return

    # Blokir SparseDataFrame deprecated
    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = (
            "SparseDataFrame sudah kedaluwarsa. Gunakan DataFrame biasa dengan kolom sparse."
        )
        raise TypeError(msg)

    if df.size == 0:
        return
    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Pembatasan Pandas: nama kolom bertipe int harus diawali 0 atau ubah jadi string: "
                "`df.columns = [str(i) for i in df.columns]`."
            )

    # Cek tipe data: semua boolean (ataboleh NaN bila null_values=True)
    if null_values:
        all_bools = (
            df.apply(lambda col: col.apply(lambda x: pd.isna(x) or isinstance(x, bool)))
            .all()
            .all()
        )
    else:
        all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()

    if not all_bools:
        warnings.warn(
            "Tipe data non-bool memperlambat komputasi. "
            "Gunakan DataFrame bertipe bool.",
            DeprecationWarning,
        )

        has_nans = pd.isna(df).any().any()
        if null_values and not has_nans:
            warnings.warn(
                "null_values=True lambat bila tak ada NaN. Gunakan False."
            )
        if not null_values and has_nans:
            raise ValueError(
                "NaN values are not permitted"
            )

        # Ambil nilai aktual untuk dicek
        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values

        if null_values:
            idxs = np.where((values != 1) & (values != 0) & (~np.isnan(values)))
        else:
            idxs = np.where((values != 1) & (values != 0))

        if len(idxs[0]) > 0:
            val = values[tuple(loc[0] for loc in idxs)]
            s = (
                "The allowed values for a DataFrame"
                " are True, False, 0, 1. Found value %s" % (val)
            )

            if null_values:
                s = (
                    "The allowed values for a DataFrame"
                    " are True, False, 0, 1, NaN. Found value %s" % (val)
                )
            raise ValueError(s)

def fpgrowth(
    df, min_support=0.5, null_values=False, use_colnames=False, max_len=None, verbose=0
):
    # Langkah 1: Validasi DataFrame masukan dan flag penanganan nilai null
    check_valid_input(df, null_values)

    # Langkah 2: Pastikan min_support berada dalam rentang yang valid (0, 1]
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` harus angka positif "
            "dalam interval `(0, 1]`. "
            "Didapat %s." % min_support
        )

    # Langkah 3: Bangun pemetaan nama kolom jika pengguna ingin menyimpan nama kolom asli
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    # Langkah 4: Bangun FP-tree dan identifikasi item yang dinonaktifkan (di bawah min_support)
    tree, disabled, _ = setup_fptree(df, min_support)

    # Langkah 5: Ubah min_support relatif menjadi hitungan absolut
    minsup = math.ceil(min_support * len(df.index))

    # Langkah 6: Buat generator yang secara malas menghasilkan itemset yang sering
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    # Langkah 7: Pasca-proses generator untuk melampirkan nilai support dan kembalikan hasil akhir
    return generate_itemsets_with_supports(
        generator, df, disabled, min_support, len(df.index), colname_map
    )


def fpg_step(tree, minsup, colnames, max_len, verbose):
    # Langkah 1: Inisialisasi penghitung untuk itemset yang dihasilkan dan ambil item tree saat ini
    count = 0
    items = tree.nodes.keys()

    # Langkah 2: Tangani kasus khusus ketika FP-tree berupa jalur tunggal
    if tree.is_single_path():
        # Tentukan ukuran kombinasi maksimal dengan memperhatikan batasan max_len
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        # Hasilkan semua kombinasi non-kosong dari item pada jalur
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                # Kombinasi adalah hitungan minimum dari item-itemnya
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)

    # Langkah 3: Jika tree bukan jalur tunggal, hasilkan setiap perluasan item
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            # Agregasikan support untuk item di semua nodenya
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    # Langkah 4: print
    if verbose:
        tree.print_status(count, colnames)

    # Langkah 5: Secara rekursif jalankan FP-tree bersyarat untuk setiap item
    if not tree.is_single_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            # Bangun FP-tree bersyarat untuk item saat ini
            cond_tree = tree.conditional_tree(item, minsup)
            # Secara rekursif hasilkan itemset yang sering dari tree bersyarat
            for sup, iset in fpg_step(cond_tree, minsup, colnames, max_len, verbose):
                yield sup, iset
