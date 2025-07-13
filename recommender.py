import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import faiss
import itertools

# ============================== DATA PREPROCESSING ==============================

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['product_name_clean'] = df['product_name'].str.lower().str.strip()
    df['category_clean'] = df['category'].str.lower().str.strip()
    df['subcategory_clean'] = df['subcategory'].str.lower().str.strip()
    df['brand_clean'] = df['brand'].str.lower().str.strip()
    df['product_text'] = df['product_name_clean'] + ' ' + df['category_clean'] + ' ' + df['subcategory_clean'] + ' ' + df['brand_clean']
    df['price_bin'] = pd.cut(df['price'], bins=10, labels=False)
    df['revenue'] = df['price'] * df['quantity']
    return df

def create_baskets(df):
    baskets = df.groupby(['customer_id', 'date'])['product_id'].apply(list).reset_index()
    baskets['basket_size'] = baskets['product_id'].apply(len)
    return baskets[baskets['basket_size'] > 1]

# ============================== FEATURE EXTRACTION ==============================

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()
        
    def extract_text_features(self, df, method='tfidf'):
        if method == 'tfidf':
            return self.tfidf_vectorizer.fit_transform(df['product_text']).toarray()
        elif method == 'bert':
            return self.bert_model.encode(df['product_text'].tolist())

    def extract_metadata_features(self, df):
        cat_features = pd.get_dummies(df[['category', 'subcategory', 'brand', 'store_location']])
        num_features = self.scaler.fit_transform(df[['price', 'price_bin']])
        return np.hstack([cat_features.values, num_features])

# ============================== SIMILAR PRODUCT RECOMMENDER ==============================

class SimilarProductRecommender:
    def __init__(self, use_faiss=True):
        self.use_faiss = use_faiss
        self.similarity_matrix = None
        self.faiss_index = None
        self.product_features = None
        
    def fit(self, df, text_method='bert', alpha=0.7):
        extractor = FeatureExtractor()
        text_features = extractor.extract_text_features(df, method=text_method)
        metadata_features = extractor.extract_metadata_features(df)
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        metadata_features = metadata_features / np.linalg.norm(metadata_features, axis=1, keepdims=True)
        self.product_features = np.hstack([
            alpha * text_features,
            (1 - alpha) * metadata_features
        ])
        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.product_features.shape[1])
            self.faiss_index.add(self.product_features.astype('float32'))
        else:
            self.similarity_matrix = cosine_similarity(self.product_features)
    
    def recommend(self, product_idx, n_recommendations=5):
        if self.use_faiss:
            query = self.product_features[product_idx:product_idx+1].astype('float32')
            sims, idxs = self.faiss_index.search(query, n_recommendations + 1)
            return list(zip(idxs[0][1:], sims[0][1:]))
        else:
            sims = self.similarity_matrix[product_idx]
            idxs = np.argsort(sims)[::-1][1:n_recommendations+1]
            return list(zip(idxs, sims[idxs]))

# ============================== FREQUENTLY BOUGHT TOGETHER ==============================

class FrequentlyBoughtTogetherRecommender:
    def __init__(self, method='apriori'):
        self.method = method
        self.rules = None
        self.product_cooccurrence = None
        
    def fit(self, baskets, min_support=0.01, min_confidence=0.5):
        transactions = baskets['product_id'].tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        self.rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
        self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=False)
    
    def recommend(self, product_id, n_recommendations=5):
        rel_rules = self.rules[self.rules['antecedents'].apply(lambda x: product_id in x)]
        results = []
        for _, row in rel_rules.head(n_recommendations).iterrows():
            for item in row['consequents']:
                results.append((item, row['confidence']))
        return results[:n_recommendations]

# ============================== STREAMLIT UI ==============================

# Cached function to load and fit the Similar Product Recommender
@st.cache_resource
def get_similar_recommender(df):
    print("Fitting Similar Product Recommender...") # This will print only on first run
    recommender = SimilarProductRecommender(use_faiss=True)
    # Fit on unique products to avoid redundant calculations and ensure correct indexing
    unique_products_df = df.drop_duplicates(subset=['product_id']).reset_index(drop=True)
    recommender.fit(unique_products_df, text_method='bert', alpha=0.7)
    return recommender, unique_products_df

# Cached function to load and fit the FBT Recommender
@st.cache_resource
def get_fbt_recommender(df):
    print("Fitting FBT Recommender...") # This will print only on first run
    baskets = create_baskets(df)
    if baskets.empty or len(baskets) < 10: # Add a check for minimum baskets
        return None
    fbt = FrequentlyBoughtTogetherRecommender()
    # Using lower thresholds to ensure recommendations are found in a sample dataset
    fbt.fit(baskets, min_support=0.005, min_confidence=0.1) 
    return fbt

def main():
    st.set_page_config(page_title="Walmart Recommender", layout="wide")

    # --- 1. DATA & MODEL LOADING ---
    @st.cache_data
    def load_data():
        df = pd.read_csv("product_orders.csv")
        return preprocess_data(df)

    df = load_data()
    similar_recommender, unique_products_df = get_similar_recommender(df)
    fbt_recommender = get_fbt_recommender(df)

    # --- Initialize session state for tracking user interactions ---
    if 'show_recs_for' not in st.session_state:
        st.session_state['show_recs_for'] = None

    # ============================ UI LAYOUT ============================

    st.title("Advanced Product Recommender")
    st.markdown("---")

    # --- a] Search for items in the start ---
    st.header("Search for a Product")
    product_list = [""] + sorted(unique_products_df['product_name'].unique().tolist())
    selected_product_name = st.selectbox(
        "Start typing to find a product...",
        product_list,
        label_visibility="collapsed"
    )

    # Display recommendations if a product is selected from search
    if selected_product_name:
        st.subheader(f"Recommendations for: {selected_product_name}")
        product_info = unique_products_df[unique_products_df['product_name'] == selected_product_name].iloc[0]
        product_idx = product_info.name  # .name gets the index in unique_products_df
        
        sim_col, fbt_col = st.columns(2)
        with sim_col:
            st.markdown("##### More in the category")
            recommendations = similar_recommender.recommend(product_idx, n_recommendations=3)
            for rec_idx, score in recommendations:
                rec_product = unique_products_df.iloc[rec_idx]
                st.info(f"{rec_product['product_name']} (${rec_product['price']:.2f})")
        with fbt_col:
            st.markdown("##### Frequently Bought Together")
            if fbt_recommender:
                fbt_recs = fbt_recommender.recommend(product_info['product_id'], n_recommendations=3)
                if fbt_recs:
                    for rec_id, conf in fbt_recs:
                        rec_product_series = unique_products_df[unique_products_df['product_id'] == rec_id]
                        if not rec_product_series.empty:
                            rec_product = rec_product_series.iloc[0]
                            st.success(f"{rec_product['product_name']} (${rec_product['price']:.2f})")
                else:
                    st.warning("No 'Bought Together' recommendations found.")
    st.markdown("---")


    # --- b] Top 10 trending items ---
    st.header("Top 10 Trending Items")
    # Define "trending" as highest total quantity sold
    trending_df = df.groupby(['product_id', 'product_name', 'brand', 'price', 'category'])['quantity'].sum().reset_index()
    top_10_trending = trending_df.sort_values(by='quantity', ascending=False).head(10)

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(top_10_trending):
                product = top_10_trending.iloc[i + j]
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(f"**{product['product_name']}**")
                        st.caption(f"{product['brand']} | ${product['price']:.2f}")
                        with st.expander("Show similar"):
                            product_info = unique_products_df[unique_products_df['product_id'] == product['product_id']]
                            if not product_info.empty:
                                product_idx = product_info.index[0]
                                recommendations = similar_recommender.recommend(product_idx, n_recommendations=3)
                                for rec_idx, score in recommendations:
                                    rec_product = unique_products_df.iloc[rec_idx]
                                    st.write(f"_{rec_product['product_name']}_")
    st.markdown("---")

    # --- c] Filtering products according to category and brand ---
    st.header("Browse & Discover")
    
    cat_col, brand_col = st.columns(2)
    with cat_col:
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("1. Filter by Category", categories)
    with brand_col:
        if selected_category == 'All':
            brands = ['All'] + sorted(df['brand'].unique().tolist())
        else:
            brands = ['All'] + sorted(df[df['category'] == selected_category]['brand'].unique().tolist())
        selected_brand = st.selectbox("2. Filter by Brand", brands)

    # Filter the dataframe of unique products for display
    filtered_display_df = unique_products_df.copy()
    if selected_category != 'All':
        filtered_display_df = filtered_display_df[filtered_display_df['category'] == selected_category]
    if selected_brand != 'All':
        filtered_display_df = filtered_display_df[filtered_display_df['brand'] == selected_brand]

    if not filtered_display_df.empty:
        st.write(f"Showing {len(filtered_display_df)} products...")
        cols_per_row = 4
        for i in range(0, len(filtered_display_df), cols_per_row):
            cols = st.columns(cols_per_row)
            row_products = filtered_display_df.iloc[i : i + cols_per_row]
            
            for idx, product in row_products.iterrows():
                col_index = (idx - i) % cols_per_row
                with cols[col_index]:
                    with st.container(border=True, height=200):
                        st.markdown(f"**{product['product_name']}**")
                        st.caption(f"{product['brand']} | ${product['price']:.2f}")
                        
                        button_key = f"add_{product['product_id']}"
                        if st.button("Recommendation more", key=button_key, use_container_width=True):
                            if st.session_state.get('show_recs_for') == product['product_id']:
                                st.session_state['show_recs_for'] = None
                            else:
                                st.session_state['show_recs_for'] = product['product_id']
                            st.rerun()

            # Display recommendations below the row if a product in that row was selected
            product_id_to_show_recs = st.session_state.get('show_recs_for')
            if product_id_to_show_recs and product_id_to_show_recs in row_products['product_id'].values:
                with st.container(border=True):
                    selected_product_info = unique_products_df[unique_products_df['product_id'] == product_id_to_show_recs].iloc[0]
                    st.subheader(f"Recommendations for {selected_product_info['product_name']}")
                    rec_col1, rec_col2 = st.columns(2)

                    with rec_col1:
                        st.markdown("##### More in the category")
                        product_idx = selected_product_info.name
                        sim_recs = similar_recommender.recommend(product_idx, n_recommendations=3)
                        for rec_idx, score in sim_recs:
                            st.info(f"{unique_products_df.iloc[rec_idx]['product_name']}")

                    with rec_col2:
                        st.markdown("##### Frequently Bought Together")
                        if fbt_recommender:
                            fbt_recs = fbt_recommender.recommend(product_id_to_show_recs, n_recommendations=2)
                            if fbt_recs:
                                for rec_id, conf in fbt_recs:
                                    rec_prod_series = unique_products_df[unique_products_df['product_id'] == rec_id]
                                    if not rec_prod_series.empty:
                                        st.success(f"{rec_prod_series.iloc[0]['product_name']}")
                            else:
                                st.warning("No recommendations found.")
    else:
        st.warning("No products match the selected filters.")

if __name__ == "__main__":
    main()