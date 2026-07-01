import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def main():
    st.set_page_config(page_title="Correlation Heatmap Generator", layout="wide")
    
    st.title("Correlation Heatmap Generator")
    st.markdown("Upload your raw data CSV to dynamically categorize and analyze Pearson correlations.")

    # Sidebar Controls
    st.sidebar.header("Visualization Controls")
    
    custom_title = st.sidebar.text_input("Heatmap Title", value="Pearson Correlation Heatmap")
    
    cmap_selection = st.sidebar.selectbox(
        "Color Palette", 
        options=["vlag", "coolwarm", "Spectral", "icefire", "RdYlBu_r"],
        index=0
    )
    
    fig_width = st.sidebar.slider("Figure Width", min_value=6, max_value=24, value=12)
    fig_height = st.sidebar.slider("Figure Height", min_value=6, max_value=24, value=10)
    
    show_values = st.sidebar.toggle("Show Values inside Heatmap", value=True)
    
    annot_font_size = st.sidebar.slider("Annotation Font Size", min_value=4, max_value=16, value=9)

    # Main Area
    uploaded_file = st.file_uploader("Upload Raw Data CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Data Ingestion
            df = pd.read_csv(uploaded_file)
            df.set_index(df.columns[0], inplace=True)
            
            # Clean numeric data
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric.dropna(axis=1, how='all', inplace=True)
            
            available_cols = df_numeric.columns.tolist()
            
            # Dynamic Column Categorization
            st.markdown("### Categorize Columns")
            
            expected_y = [
                'Bifidobacterium', 'Lactobacillus', 'Faecalibacterium', 
                'Enterobacterium', 'Enterococcus', 'Bacteroidetes', 
                'C leptum', 'C coccoides', 'Prevotella'
            ]
            expected_x = [
                'Total SCFA', 'Acetic acid', 'Butyric acid', 
                'Propionic acid', 'Valeric acid'
            ]
            
            default_y = [col for col in expected_y if col in available_cols]
            default_x = [col for col in expected_x if col in available_cols]
            
            col1, col2 = st.columns(2)
            with col1:
                y_axis_cols = st.multiselect(
                    "Y-Axis (Rows)", 
                    options=available_cols, 
                    default=default_y
                )
            with col2:
                x_axis_cols = st.multiselect(
                    "X-Axis (Columns)", 
                    options=available_cols, 
                    default=default_x
                )
            
            swap_axes = st.toggle("🔄 Swap X and Y Axes")
            
            st.markdown("---")
            
            # Mathematical Processing
            corr_matrix = df_numeric.corr(method='pearson')
            
            # Tabbed Interface
            tab1, tab2 = st.tabs(["Targeted Correlation", "Full Correlation Matrix"])
            
            with tab1:
                if not y_axis_cols or not x_axis_cols:
                    st.warning("Please select at least one column for both the X and Y axes.")
                else:
                    st.subheader("Targeted Heatmap")
                    
                    if swap_axes:
                        sub_corr = corr_matrix.loc[x_axis_cols, y_axis_cols]
                    else:
                        sub_corr = corr_matrix.loc[y_axis_cols, x_axis_cols]
                    
                    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                    ax1.set_title(custom_title, pad=20, fontsize=16)
                    
                    sns.heatmap(
                        sub_corr,
                        vmin=-1.0,
                        vmax=1.0,
                        cmap=cmap_selection, 
                        annot=show_values,
                        fmt=".2f", 
                        linewidths=.5, 
                        cbar_kws={"shrink": .8}, 
                        ax=ax1, 
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
                
                full_corr_display = corr_matrix.T if swap_axes else corr_matrix
                
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                ax2.set_title(custom_title, pad=20, fontsize=16)
                
                sns.heatmap(
                    full_corr_display,
                    vmin=-1.0,
                    vmax=1.0,
                    cmap=cmap_selection, 
                    annot=show_values,
                    fmt=".2f", 
                    linewidths=.5, 
                    cbar_kws={"shrink": .8}, 
                    ax=ax2, 
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
                c4.download_button("Download Full Matrix Data (CSV)", data=full_corr_display.to_csv().encode('utf-8'), file_name="full_correlations.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()