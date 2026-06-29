import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def main():
    st.set_page_config(page_title="Microbiome Heatmap Generator", layout="wide")
    
    st.title("Microbiome & Metabolite Correlation")
    st.markdown("Upload your raw data CSV to dynamically categorize and analyze Pearson correlations.")

    # Sidebar Controls
    st.sidebar.header("Visualization Controls")
    cmap_selection = st.sidebar.selectbox(
        "Color Palette", 
        options=["vlag", "coolwarm", "Spectral", "icefire", "RdYlBu_r"],
        index=0
    )
    
    fig_width = st.sidebar.slider("Figure Width", min_value=6, max_value=24, value=12)
    fig_height = st.sidebar.slider("Figure Height", min_value=6, max_value=24, value=10)
    annot_font_size = st.sidebar.slider("Annotation Font Size", min_value=4, max_value=16, value=9)

    # Main Area
    uploaded_file = st.file_uploader("Upload Raw Data CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # 1. Data Ingestion
            df = pd.read_csv(uploaded_file)
            df.set_index(df.columns[0], inplace=True)
            
            # Clean numeric data
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric.dropna(axis=1, how='all', inplace=True)
            
            available_cols = df_numeric.columns.tolist()
            
            # 2. Dynamic Column Categorization
            st.markdown("### Categorize Columns")
            
            # Pre-defined lists for auto-population (if they exist in the file)
            expected_bacteria = [
                'Bifidobacterium', 'Lactobacillus', 'Faecalibacterium', 
                'Enterobacterium', 'Enterococcus', 'Bacteroidetes', 
                'C leptum', 'C coccoides', 'Prevotella'
            ]
            expected_metabolites = [
                'Total SCFA', 'Acetic acid', 'Butyric acid', 
                'Propionic acid', 'Valeric acid'
            ]
            
            default_group1 = [col for col in expected_bacteria if col in available_cols]
            default_group2 = [col for col in expected_metabolites if col in available_cols]
            
            col1, col2 = st.columns(2)
            with col1:
                group1_cols = st.multiselect(
                    "Group 1 (e.g., Bacteria - Rows)", 
                    options=available_cols, 
                    default=default_group1
                )
            with col2:
                group2_cols = st.multiselect(
                    "Group 2 (e.g., Metabolites - Columns)", 
                    options=available_cols, 
                    default=default_group2
                )
            
            st.markdown("---")
            
            # 3. Mathematical Processing
            corr_matrix = df_numeric.corr(method='pearson')
            
            # 4. Tabbed Interface
            tab1, tab2 = st.tabs(["Targeted Correlation (Group 1 vs Group 2)", "Full Correlation Matrix"])
            
            with tab1:
                if not group1_cols or not group2_cols:
                    st.warning("Please select at least one column for both Group 1 and Group 2 to generate this heatmap.")
                else:
                    st.subheader("Targeted Heatmap")
                    sub_corr = corr_matrix.loc[group1_cols, group2_cols]
                    
                    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                    sns.heatmap(
                        sub_corr, cmap=cmap_selection, annot=True, fmt=".2f", 
                        linewidths=.5, cbar_kws={"shrink": .8}, ax=ax1, 
                        annot_kws={"size": annot_font_size}
                    )
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig1)
                    
                    # Export for Targeted Matrix
                    img_buffer1 = io.BytesIO()
                    fig1.savefig(img_buffer1, format="png", bbox_inches="tight", dpi=300)
                    
                    c1, c2 = st.columns(2)
                    c1.download_button("Download Targeted Heatmap (PNG)", data=img_buffer1.getvalue(), file_name="targeted_heatmap.png", mime="image/png")
                    c2.download_button("Download Targeted Data (CSV)", data=sub_corr.to_csv().encode('utf-8'), file_name="targeted_correlations.csv", mime="text/csv")
            
            with tab2:
                st.subheader("Full Matrix Heatmap")
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(
                    corr_matrix, cmap=cmap_selection, annot=True, fmt=".2f", 
                    linewidths=.5, cbar_kws={"shrink": .8}, ax=ax2, 
                    annot_kws={"size": annot_font_size}
                )
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig2)
                
                # Export for Full Matrix
                img_buffer2 = io.BytesIO()
                fig2.savefig(img_buffer2, format="png", bbox_inches="tight", dpi=300)
                
                c3, c4 = st.columns(2)
                c3.download_button("Download Full Heatmap (PNG)", data=img_buffer2.getvalue(), file_name="full_heatmap.png", mime="image/png")
                c4.download_button("Download Full Matrix Data (CSV)", data=corr_matrix.to_csv().encode('utf-8'), file_name="full_correlations.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()