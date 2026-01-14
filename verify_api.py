import requests

def test_analyze():
    url = "http://localhost:8000/analyze"
    files = {'file': open('market_basket_dataset.csv', 'rb')}
    data = {
        'max_iter': 100,
        'tabu_size': 18,
        'k_focus_items': 50,
        'tabu_threshold': 0.1,
        'min_support': 0.001,
        'min_confidence': 0.2
    }
    
    print("Sending request to /analyze...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print("Success!")
        results = response.json()
        print(f"Selected focus items: {results['selected_focus_items']}")
        print(f"Number of frequent itemsets found: {len(results['frequent_itemsets'])}")
        print(f"Number of cross-selling strategies found: {len(results['cross_selling_strategies'])}")
        
        # Verify first few itemsets
        if results['frequent_itemsets']:
            print("\nFirst 5 frequent itemsets:")
            for item in results['frequent_itemsets'][:5]:
                print(f"  {item['itemset']}: {item['support']:.4f}")
        
        # Verify first few strategies
        if results['cross_selling_strategies']:
            print("\nTop 5 cross-selling strategies (by lift):")
            for rule in results['cross_selling_strategies'][:5]:
                print(f"  {rule['antecedents']} -> {rule['consequents']} (lift: {rule['lift']:.4f}, confidence: {rule['confidence']:.4f})")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_analyze()
