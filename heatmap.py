import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import io

def calculate_pvalues(df, method='pearson'):
    """Helper function to calculate a p-value matrix for a dataframe."""
    cols = df.columns
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for r in cols:
        for c in cols:
            if r == c:
                pvals.loc[r, c] = 0.0 
                continue
                
            mask = df[r].notna() & df[c].notna()
            x, y = df[r][mask], df[c][mask]
            
            if len(x) < 2:
                pvals.loc[r, c] = np.nan
                continue
                
            try:
                if method == 'pearson':
                    _, p = stats.pearsonr(x, y)
                elif method == 'spearman':
                    _, p = stats.spearmanr(x, y)
                elif method == 'kendall':
                    _, p = stats.kendalltau(x, y)
                pvals.loc[r, c] = p
            except:
                pvals.loc[r, c] = np.nan
                
    return pvals

def main():
    st.set_page_config(page_title="Correlation Heatmap Generator", layout="wide")
    
    st.title("Correlation Heatmap Generator")
    st.markdown("Upload your raw data CSV to dynamically categorize and analyze correlations.")

    # Sidebar Controls
    st.sidebar.header("Heatmap Controls")

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
            outlier_filter = st.sidebar.number_input(
                "Coefficient of Variation %", 
                min_value=0, 
                max_value=None, 
                value=15
            )
    
    with st.sidebar.expander("Title & Axis Labels"):
        custom_title = st.text_input("Heatmap Title", value="Correlation Heatmap")
        y_axis_label = st.text_input("Y-Axis Label", value="Bacteria")
        x_axis_label = st.text_input("X-Axis Label", value="Metabolites")
    
    with st.sidebar.expander("Heatmap Appearance"):    
        cmap_selection = st.selectbox(
            "Color Palette", 
            options=["vlag", "coolwarm", "Spectral", "icefire", "RdYlBu_r"],
            index=0
        )
        
        fig_width = st.slider("Figure Width", min_value=6, max_value=24, value=12)
        fig_height = st.slider("Figure Height", min_value=6, max_value=24, value=10)
        annot_font_size = st.slider("Annotation Font Size", min_value=4, max_value=16, value=9)
        show_values = st.toggle("Show Values inside Heatmap", value=True)
        mask_option = st.selectbox(
            "Matrix Masking for Full Matrix",
            options=["None", "Hide Upper Triangle", "Hide Lower Triangle"],
            index=0
        )    

    with st.sidebar.expander("Significance Filtering"):
        sig_metric = st.radio(
            "Threshold Metric",
            options=["P-value (α)", "Correlation Coefficient (|r|)"]
        )
    
        if sig_metric == "Correlation Coefficient (|r|)":
            sig_threshold = st.slider("Threshold (|r| ≥)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        else:
            st.markdown("**P-Value Tiers (α ≤)**")
            p_star = st.number_input("* Threshold", value=0.10, step=0.01)
            p_2star = st.number_input("** Threshold", value=0.05, step=0.01)
            p_3star = st.number_input("*** Threshold", value=0.001, step=0.001, format="%.3f")
            sig_threshold = p_star 
        
        sig_action = st.radio(
            "Threshold Action", 
            options=["Highlight Significant (*)", "Mask Insignificant Cells"]
        )

    # Main Area
    uploaded_file = st.file_uploader("Upload Raw Data CSV (Headers in Row 1)", type=["csv"])

    if uploaded_file is not None:
        try:
            # 1. Data Ingestion
            df = pd.read_csv(uploaded_file)
            df.set_index(df.columns[0], inplace=True)
            
            # Clean numeric data
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric.dropna(axis=1, how='all', inplace=True)
            
            # 2. Replicate Handling Logic
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
                                if abs(cv) > outlier_filter / 100: 
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
                        st.markdown("The following technical replicates exceeded a 15% Coefficient of Variation. The outlier was dropped before calculating the mean.")
                        for log in outlier_logs:
                            st.markdown(f"- {log}")
                    with st.expander(f"📃 Final DataFrame after Outlier Filtering"):
                        st.table(df_numeric)
            
            available_cols = df_numeric.columns.tolist()
            
            # 3. Dynamic Column Categorization
            st.markdown("### Categorize Columns")
            
            expected_bacteria = [
                'Bifidobacterium', 'Lactobacillus', 'Faecalibacterium', 
                'Enterobacterium', 'Enterococcus', 'Bacteroidetes', 
                'C leptum', 'C coccoides', 'Prevotella'
            ]
            expected_metabolites = [
                'Total SCFA', 'Acetic acid', 'Butyric acid', 
                'Propionic acid', 'Valeric acid'
            ]
            
            default_y = [col for col in expected_bacteria if col in available_cols]
            default_x = [col for col in expected_metabolites if col in available_cols]
            
            col1, col2 = st.columns(2)
            with col1:
                y_axis_cols = st.multiselect("Y-Axis (Rows)", options=available_cols, default=default_y)
            with col2:
                x_axis_cols = st.multiselect("X-Axis (Columns)", options=available_cols, default=default_x)
            
            swap_axes = st.toggle("🔄 Swap X and Y Axes")
            
            st.markdown("---")
            
            # 4. Mathematical Processing
            corr_matrix = df_numeric.corr(method=corr_method)
            pval_matrix = calculate_pvalues(df_numeric, method=corr_method)
            
            # 5. Tabbed Interface
            tab1, tab2 = st.tabs(["Targeted Correlation", "Full Correlation Matrix"])
            
            with tab1:
                if not y_axis_cols or not x_axis_cols:
                    st.warning("Please select at least one column for both the X and Y axes.")
                else:
                    st.subheader("Targeted Heatmap")
                    
                    if swap_axes:
                        sub_corr = corr_matrix.loc[x_axis_cols, y_axis_cols]
                        sub_pval = pval_matrix.loc[x_axis_cols, y_axis_cols]
                    else:
                        sub_corr = corr_matrix.loc[y_axis_cols, x_axis_cols]
                        sub_pval = pval_matrix.loc[y_axis_cols, x_axis_cols]
                        
                    # FIXED: Explicit shape definition to prevent transposed memory bugs
                    annot_matrix1 = np.empty(sub_corr.shape, dtype=object)
                    
                    if sig_metric == "Correlation Coefficient (|r|)":
                        sig_mask1 = np.abs(sub_corr) < sig_threshold
                    else:
                        sig_mask1 = (sub_pval > sig_threshold) | sub_pval.isna()
                    
                    for i in range(sub_corr.shape[0]):
                        for j in range(sub_corr.shape[1]):
                            val = sub_corr.iloc[i, j]
                            base_text = f"{val:.2f}" if show_values else ""
                            
                            if sig_action == "Highlight Significant (*)":
                                if sig_metric == "Correlation Coefficient (|r|)":
                                    annot_matrix1[i, j] = f"{base_text}*" if abs(val) >= sig_threshold else base_text
                                else:
                                    p_val = sub_pval.iloc[i, j]
                                    if pd.notna(p_val):
                                        if p_val <= p_3star:
                                            stars = "***"
                                        elif p_val <= p_2star:
                                            stars = "**"
                                        elif p_val <= p_star:
                                            stars = "*"
                                        else:
                                            stars = ""
                                        annot_matrix1[i, j] = f"{base_text}{stars}"
                                    else:
                                        annot_matrix1[i, j] = base_text
                            else:
                                annot_matrix1[i, j] = base_text
                                
                    final_mask1 = sig_mask1 if sig_action == "Mask Insignificant Cells" else None
                    
                    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                    ax1.set_title(custom_title, pad=20, fontsize=16)
                    
                    sns.heatmap(
                        sub_corr, mask=final_mask1, cmap=cmap_selection, vmin=-1.0, vmax=1.0,  
                        annot=annot_matrix1, fmt="", linewidths=.5, 
                        cbar_kws={"shrink": .8}, ax=ax1, annot_kws={"size": annot_font_size}
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
                    
                    img_buffer1 = io.BytesIO()
                    fig1.savefig(img_buffer1, format="png", bbox_inches="tight", dpi=300)
                    c1, c2, c3 = st.columns(3)
                    c1.download_button("Download Heatmap (PNG)", data=img_buffer1.getvalue(), file_name="targeted_heatmap.png", mime="image/png")
                    c2.download_button(f"Download Correlations (CSV)", data=sub_corr.to_csv().encode('utf-8'), file_name=f"targeted_correlations_{corr_method}.csv", mime="text/csv")
                    c3.download_button(f"Download P-Values (CSV)", data=sub_pval.to_csv().encode('utf-8'), file_name=f"targeted_pvalues_{corr_method}.csv", mime="text/csv")
            
            with tab2:
                st.subheader("Full Matrix Heatmap")
                
                full_corr_display = corr_matrix.T if swap_axes else corr_matrix
                full_pval_display = pval_matrix.T if swap_axes else pval_matrix
                
                # FIXED: Explicit shape definition to prevent transposed memory bugs
                annot_matrix2 = np.empty(full_corr_display.shape, dtype=object)
                
                if sig_metric == "Correlation Coefficient (|r|)":
                    sig_mask2 = np.abs(full_corr_display) < sig_threshold
                else:
                    sig_mask2 = (full_pval_display > sig_threshold) | full_pval_display.isna()
                
                for i in range(full_corr_display.shape[0]):
                    for j in range(full_corr_display.shape[1]):
                        val = full_corr_display.iloc[i, j]
                        base_text = f"{val:.2f}" if show_values else ""
                        
                        if sig_action == "Highlight Significant (*)":
                            if sig_metric == "Correlation Coefficient (|r|)":
                                annot_matrix2[i, j] = f"{base_text}*" if abs(val) >= sig_threshold else base_text
                            else:
                                p_val = full_pval_display.iloc[i, j]
                                if pd.notna(p_val):
                                    if p_val <= p_3star:
                                        stars = "***"
                                    elif p_val <= p_2star:
                                        stars = "**"
                                    elif p_val <= p_star:
                                        stars = "*"
                                    else:
                                        stars = ""
                                    annot_matrix2[i, j] = f"{base_text}{stars}"
                                else:
                                    annot_matrix2[i, j] = base_text
                        else:
                            annot_matrix2[i, j] = base_text
                
                structural_mask = None
                if mask_option == "Hide Upper Triangle":
                    # FIXED: Explicit shape generation
                    structural_mask = np.triu(np.ones(full_corr_display.shape, dtype=bool))
                elif mask_option == "Hide Lower Triangle":
                    # FIXED: Explicit shape generation
                    structural_mask = np.tril(np.ones(full_corr_display.shape, dtype=bool))
                
                if sig_action == "Mask Insignificant Cells":
                    final_mask2 = structural_mask | sig_mask2 if structural_mask is not None else sig_mask2
                else:
                    final_mask2 = structural_mask
                
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                ax2.set_title(custom_title, pad=20, fontsize=16)
                
                sns.heatmap(
                    full_corr_display, mask=final_mask2, cmap=cmap_selection, vmin=-1.0, vmax=1.0,  
                    annot=annot_matrix2, fmt="", linewidths=.5, 
                    cbar_kws={"shrink": .8}, ax=ax2, annot_kws={"size": annot_font_size}
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
                
                img_buffer2 = io.BytesIO()
                fig2.savefig(img_buffer2, format="png", bbox_inches="tight", dpi=300)
                c1, c2, c3 = st.columns(3)
                c1.download_button("Download Heatmap (PNG)", data=img_buffer2.getvalue(), file_name="full_heatmap.png", mime="image/png")
                c2.download_button(f"Download Correlations (CSV)", data=full_corr_display.to_csv().encode('utf-8'), file_name=f"full_correlations_{corr_method}.csv", mime="text/csv")
                c3.download_button(f"Download P-Values (CSV)", data=full_pval_display.to_csv().encode('utf-8'), file_name=f"full_pvalues_{corr_method}.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()