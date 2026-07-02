import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def main():
    st.set_page_config(page_title="Correlation Heatmap Generator", layout="wide")
    
    st.title("Correlation Heatmap Generator")
    st.markdown("Upload your raw data CSV to dynamically categorize and analyze correlations.")

    # Sidebar Controls
    st.sidebar.header("Visualization Controls")
    
    # Correlation Method Selector
    corr_method = st.sidebar.selectbox(
        "Correlation Method",
        options=["pearson", "spearman", "kendall"],
        format_func=lambda x: x.capitalize(),
        index=0
    )

    replicate_handling = st.sidebar.selectbox(
        "Replicate Handling",
        options=["None", "Median", "Mean"],
        index=0
    )

    if replicate_handling == "Mean":
        outlier_filter_checkbox =st.sidebar.checkbox("Enable Outlier Filtering", value=True)
        if outlier_filter_checkbox:
            outlier_filter = st.sidebar.slider(
                "Outlier Filter Threshold (Coefficient of Variation %)", 
                min_value=0, 
                max_value=100, 
                value=15, 
                step=1
            )
    
    custom_title = st.sidebar.text_input("Heatmap Title", value="Correlation Heatmap")
    y_axis_label = st.sidebar.text_input("Y-Axis Label", value="Bacteria")
    x_axis_label = st.sidebar.text_input("X-Axis Label", value="Metabolites")
    
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
            # 1. Data Ingestion
            df = pd.read_csv(uploaded_file)
            df.set_index(df.columns[0], inplace=True)
            
            # Clean numeric data
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric.dropna(axis=1, how='all', inplace=True)
            
            if replicate_handling == "Median":
                df_numeric = df_numeric.groupby(level=0).median()
                
            elif replicate_handling == "Mean" and outlier_filter_checkbox:
                outlier_logs = []
                processed_frames = []
                
                for sample_name, group in df_numeric.groupby(level=0):
                    if len(group) >= 3:
                        sample_res = {}
                        for col in group.columns:
                            vals = group[col].dropna()
                            if len(vals) >= 3:
                                cv = vals.std(ddof=1) / vals.mean() if vals.mean() != 0 else 0
                                if abs(cv) > (outlier_filter / 100):
                                    vals_reset = vals.reset_index(drop=True)
                                    med = vals_reset.median()
                                    outlier_idx = (vals_reset - med).abs().idxmax()
                                    vals_cleaned = vals_reset.drop(outlier_idx)
                                    
                                    sample_res[col] = vals_cleaned.mean()
                                    outlier_logs.append(f"**{sample_name}** - {col} (CV: {abs(cv)*100:.1f}%)")
                                else:
                                    sample_res[col] = vals.mean()
                            else:
                                sample_res[col] = vals.mean()
                        processed_frames.append(pd.DataFrame([sample_res], index=[sample_name]))
                    else:
                        processed_frames.append(pd.DataFrame([group.mean()], index=[sample_name]))
                        
                df_numeric = pd.concat(processed_frames)
                
                if outlier_logs:
                    with st.expander(f"⚠️ Outliers filtered in {len(outlier_logs)} measurements"):
                        st.markdown(f"The following technical replicates exceeded a {outlier_filter}% Coefficient of Variation. The outlier was dropped before calculating the mean.")
                        for log in outlier_logs:
                            st.markdown(f"- {log}")
                    
                        st.table(df_numeric)
            
            available_cols = df_numeric.columns.tolist()
            
            # 2. Dynamic Column Categorization
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
                    f"Y-Axis (Rows): {y_axis_label}", 
                    options=available_cols, 
                    default=default_y
                )
            with col2:
                x_axis_cols = st.multiselect(
                    f"X-Axis (Columns): {x_axis_label}", 
                    options=available_cols, 
                    default=default_x
                )
            
            swap_axes = st.toggle("🔄 Swap X and Y Axes")
            
            st.markdown("---")
            
            # 3. Mathematical Processing
            corr_matrix = df_numeric.corr(method=corr_method)
            
            # 4. Tabbed Interface
            tab1, tab2 = st.tabs(["Targeted Correlation", "Full Correlation Matrix"])
            
            with tab1:
                if not y_axis_cols or not x_axis_cols:
                    st.warning("Please select at least one column for both the X and Y axes.")
                else:
                    st.subheader(f"Targeted Heatmap - {corr_method.capitalize()} Correlation")
                    
                    if swap_axes:
                        sub_corr = corr_matrix.loc[x_axis_cols, y_axis_cols]
                    else:
                        sub_corr = corr_matrix.loc[y_axis_cols, x_axis_cols]
                    
                    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                    ax1.set_title(custom_title, pad=20, fontsize=16)
                    
                    sns.heatmap(
                        sub_corr, 
                        cmap=cmap_selection, 
                        vmin=-1.0,
                        vmax=1.0,
                        annot=show_values, 
                        fmt=".2f", 
                        linewidths=.5, 
                        cbar_kws={"shrink": .8}, 
                        ax=ax1, 
                        annot_kws={"size": annot_font_size}
                    )
                    if x_axis_label:
                        ax1.set_xlabel(x_axis_label, fontsize=12, labelpad=10)
                    else:
                        ax1.set_xlabel('')

                    if y_axis_label:
                        ax1.set_ylabel(y_axis_label, fontsize=12, labelpad=10)
                    else:
                        ax1.set_ylabel('')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig1)
                    
                    # Export for Targeted Matrix
                    img_buffer1 = io.BytesIO()
                    fig1.savefig(img_buffer1, format="png", bbox_inches="tight", dpi=300)
                    
                    c1, c2 = st.columns(2)
                    c1.download_button("Download Targeted Heatmap (PNG)", data=img_buffer1.getvalue(), file_name="targeted_heatmap.png", mime="image/png")
                    c2.download_button(f"Download Targeted Data ({corr_method.capitalize()})", data=sub_corr.to_csv().encode('utf-8'), file_name=f"targeted_correlations_{corr_method}.csv", mime="text/csv")
            
            with tab2:
                st.subheader(f"Full Matrix Heatmap - {corr_method.capitalize()} Correlation")
                
                full_corr_display = corr_matrix.T if swap_axes else corr_matrix
                
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                ax2.set_title(custom_title, pad=20, fontsize=16)
                
                sns.heatmap(
                    full_corr_display, 
                    cmap=cmap_selection, 
                    vmin=-1.0,
                    vmax=1.0,
                    annot=show_values, 
                    fmt=".2f", 
                    linewidths=.5, 
                    cbar_kws={"shrink": .8}, 
                    ax=ax2, 
                    annot_kws={"size": annot_font_size}
                )

                if x_axis_label:
                    ax2.set_xlabel(x_axis_label, fontsize=12, labelpad=10)
                else:
                    ax2.set_xlabel('')

                if y_axis_label:
                    ax2.set_ylabel(y_axis_label, fontsize=12, labelpad=10)
                else:
                    ax2.set_ylabel('')

                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig2)
                
                # Export for Full Matrix
                img_buffer2 = io.BytesIO()
                fig2.savefig(img_buffer2, format="png", bbox_inches="tight", dpi=300)
                
                c3, c4 = st.columns(2)
                c3.download_button("Download Full Heatmap (PNG)", data=img_buffer2.getvalue(), file_name="full_heatmap.png", mime="image/png")
                c4.download_button(f"Download Full Matrix Data ({corr_method.capitalize()})", data=full_corr_display.to_csv().encode('utf-8'), file_name=f"full_correlations_{corr_method}.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()