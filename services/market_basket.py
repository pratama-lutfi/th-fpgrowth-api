import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from algorithms.tabu_search import tabu_search
from algorithms.fpgrowth import fpgrowth

class MarketBasketService:
    @staticmethod
    def analyze(df, max_iter=15, tabu_size=5, k=5, threshold=0.5, min_support=0.001, min_confidence=0.2):
        # 1. Preprocessing
        # Normalize item names to lowercase
        df['Itemname'] = df['Itemname'].str.lower()
        
        # Group items by transaction (BillNo)
        transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()
        
        # 2. Tabu Search Preprocessing
        tabu_results = tabu_search(
            transactions=transactions,
            max_iter=max_iter,
            tabu_size=tabu_size,
            k=k,
            threshold=threshold
        )
        
        transactions_tabu = tabu_results['filtered_transactions']
        
        # 3. Transaction Encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions_tabu).transform(transactions_tabu)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # 4. FP-Growth
        frequent_itemsets = fpgrowth(
            df_encoded, 
            min_support=min_support, 
            use_colnames=True
        )
        
        # 5. Association Rules
        if frequent_itemsets.empty:
            rules = pd.DataFrame()
        else:
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            
        # Format Results
        frequent_itemsets_list = [
            {"itemset": list(row["itemsets"]), "support": row["support"]}
            for _, row in frequent_itemsets.iterrows()
        ]
        
        # Calculate stats
        total_transactions = len(transactions)
        total_unique_items = df['Itemname'].nunique()
        total_frequent_itemsets = len(frequent_itemsets)
        total_rules = len(rules)
        
        avg_confidence = rules['confidence'].mean() if not rules.empty else 0
        avg_lift = rules['lift'].mean() if not rules.empty else 0
        avg_support = rules['support'].mean() if not rules.empty else 0
        rules_lift_gt_2 = len(rules[rules['lift'] > 2]) if not rules.empty else 0
        rules_conf_gt_08 = len(rules[rules['confidence'] > 0.8]) if not rules.empty else 0

        # Format cross selling strategies (Top 10)
        cross_selling_strategies = []
        if not rules.empty:
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)
            for _, row in top_rules.iterrows():
                cross_selling_strategies.append({
                    "antecedents": list(row["antecedents"]),
                    "consequents": list(row["consequents"]),
                    "support": row["support"],
                    "confidence": row["confidence"],
                    "lift": row["lift"]
                })

        # Format Business Recommendations (Top 10 frequent itemsets with > 1 item)
        recommendations = []
        if not frequent_itemsets.empty:
            # Filter itemsets with more than 1 item and sort by support
            multi_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) > 1]
            top_itemsets = multi_itemsets.sort_values(by='support', ascending=False).head(10)
            for _, row in top_itemsets.iterrows():
                items_str = ", ".join(sorted(list(row["itemsets"])))
                # Assuming support in the user example is absolute count? 
                # "apples, sugar (support: 367)"
                # If so, support_count = support * total_transactions
                support_count = int(row["support"] * total_transactions)
                recommendations.append(f"   - {items_str} (support: {support_count})")

        # Format Strategi Cross-selling (Top 5 rules with lift > 5)
        cross_selling_text = []
        if not rules.empty:
            high_lift_rules = rules[rules['lift'] > 5].sort_values(by='lift', ascending=False).head(5)
            for _, rule in high_lift_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                cross_selling_text.append(f"   - Jika customer membeli {antecedents}, rekomendasikan {consequents}")
                cross_selling_text.append(f"     (Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")

        # Create the formatted summary string
        summary_str = f"""=== ANALISIS KESIMPULAN ===
Total transaksi: {total_transactions}
Total item unik: {total_unique_items}
Total frequent itemsets: {total_frequent_itemsets}
Total association rules: {total_rules}

=== STATISTIK RINGKASAN ===
Rata-rata confidence: {avg_confidence:.3f}
Rata-rata lift: {avg_lift:.3f}
Rata-rata support: {avg_support:.3f}
Jumlah rules dengan lift > 2: {rules_lift_gt_2}
Jumlah rules dengan confidence > 0.8: {rules_conf_gt_08}

=== REKOMENDASI BISNIS ===
1. Item yang sering dibeli bersama:
{chr(10).join(recommendations) if recommendations else "   (Tidak ada itemset yang memenuhi kriteria)"}

2. Strategi Cross-selling berdasarkan rules dengan lift tinggi:
{chr(10).join(cross_selling_text) if cross_selling_text else "   (Tidak ada rules dengan lift > 5)"}
"""
        
        return {
            "summary": summary_str,
            "statistics": {
                "total_transactions": total_transactions,
                "total_unique_items": total_unique_items,
                "total_frequent_itemsets": total_frequent_itemsets,
                "total_rules": total_rules,
                "avg_confidence": avg_confidence,
                "avg_lift": avg_lift,
                "avg_support": avg_support,
                "rules_lift_gt_2": rules_lift_gt_2,
                "rules_conf_gt_08": rules_conf_gt_08
            },
            "selected_focus_items": tabu_results['selected_items'],
            "frequent_itemsets": frequent_itemsets_list[:10], # Top 10 for consistency
            "cross_selling_strategies": cross_selling_strategies,
            "tabu_score": tabu_results['score']
        }
