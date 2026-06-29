import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def main():
    st.set_page_config(page_title="Heatmap Generator", layout="wide")
    
    st.title("Microbiome & Metabolite Correlation")
    st.markdown("Upload your raw data CSV to generate and customize a full Pearson correlation heatmap.")

    # Sidebar Controls
    st.sidebar.header("Visualization Controls")
    cmap_selection = st.sidebar.selectbox(
        "Color Palette", 
        options=["vlag", "coolwarm", "Spectral", "icefire", "RdYlBu_r"],
        index=0
    )
    
    fig_width = st.sidebar.slider("Figure Width", min_value=6, max_value=24, value=12)
    fig_height = st.sidebar.slider("Figure Height", min_value=6, max_value=24, value=10)
    annot_font_size = st.sidebar.slider("Annotation Font Size", min_value=4, max_value=16, value=8)

    # Main Area
    uploaded_file = st.file_uploader("Upload Raw Data CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # 1. Data Ingestion
            df = pd.read_csv(uploaded_file)
            
            # Set the first column as index (Sample IDs)
            df.set_index(df.columns[0], inplace=True)
            
            # Convert all remaining columns to numeric, coercing errors
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            
            # Drop any columns that are completely empty or non-numeric after coercion
            df_numeric.dropna(axis=1, how='all', inplace=True)
            
            # 2. Mathematical Processing (Full Matrix)
            corr_matrix = df_numeric.corr(method='pearson')
            
            # 3. Visualization Generation
            st.subheader("Pearson Correlation Heatmap")
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            sns.heatmap(
                corr_matrix, 
                cmap=cmap_selection, 
                annot=True, 
                fmt=".2f", 
                linewidths=.5, 
                cbar_kws={"shrink": .8}, 
                ax=ax,
                annot_kws={"size": annot_font_size}
            )
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            st.pyplot(fig)

            # 4. Export Capabilities
            st.markdown("---")
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            # Prepare Image Download
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
            
            with col1:
                st.download_button(
                    label="Download Heatmap (PNG)",
                    data=img_buffer.getvalue(),
                    file_name="correlation_heatmap.png",
                    mime="image/png"
                )
                
            # Prepare CSV Download
            csv_buffer = corr_matrix.to_csv().encode('utf-8')
            
            with col2:
                st.download_button(
                    label="Download Correlation Matrix (CSV)",
                    data=csv_buffer,
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.info("Please ensure the uploaded CSV matches the expected raw data structure.")

if __name__ == "__main__":
    main()