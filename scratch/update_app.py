import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add imports if not present
if 'from model.kruti_to_unicode import is_kruti_dev' not in code:
    import_target = 'LOCAL_MULTILINGUAL_NLI_MODEL_DIR = os.path.join(hf_cache_dir, "multilingual_minilmv2_l6_mnli_xnli")'
    import_replacement = (
        'LOCAL_MULTILINGUAL_NLI_MODEL_DIR = os.path.join(hf_cache_dir, "multilingual_minilmv2_l6_mnli_xnli")\n\n'
        'from model.kruti_to_unicode import is_kruti_dev, kruti_to_unicode\n'
        'from model.content_advisory import compute_parents_guide\n'
    )
    code = code.replace(import_target, import_replacement, 1)

# 2. In parse_screenplay: add kruti conversion and expand hindi_speaker_match
if 'if is_kruti_dev(content):' not in code:
    content_read_target = '    with open(file_path, \'r\', encoding=\'utf-8\', errors=\'ignore\') as f:\n        content = f.read()'
    content_read_replacement = (
        '    with open(file_path, \'r\', encoding=\'utf-8\', errors=\'ignore\') as f:\n'
        '        content = f.read()\n'
        '    if is_kruti_dev(content):\n'
        '        content = kruti_to_unicode(content)'
    )
    code = code.replace(content_read_target, content_read_replacement, 1)

old_hindi_match = "hindi_speaker_match = re.match(r'^([\\u0900-\\u097fA-Za-z][\\u0900-\\u097fA-Za-z\\s]{1,38})\\s*[:：]\\s*(.*)$', stripped)"
new_hindi_match = "hindi_speaker_match = re.match(r'^([\\u0900-\\u097fA-Za-z0-9०-९\\s\\/\\-]{1,38})\\s*[:：%]\\s*(.*)$', stripped)"
code = code.replace(old_hindi_match, new_hindi_match)

# 3. In PDF processing: auto convert Kruti Dev without triggering OCR
old_pdf_flow = """            if suffix == ".pdf":
                pdf_text = extract_pdf_text(uploaded_bytes)
                if is_garbled_text(pdf_text):
                    status.text("Legacy Hindi PDF font detected. Reading every page with Hindi OCR — this can take a few minutes on CPU...")
                    def update_ocr_status(current_page: int, total_pages: int) -> None:
                        progress_bar.progress(int(current_page / total_pages * 20))
                        status.text(f"Hindi OCR in progress: page {current_page}/{total_pages}. Please keep this tab open.")
                    pdf_text = ocr_hindi_pdf(uploaded_bytes, update_ocr_status)
                    if not pdf_text:
                        raise RuntimeError("Hindi OCR could not recover readable text from this PDF.")
                content = pdf_text.encode("utf-8")
            else:
                content = uploaded_bytes"""

new_pdf_flow = """            if suffix == ".pdf":
                pdf_text = extract_pdf_text(uploaded_bytes)
                if is_kruti_dev(pdf_text):
                    status.text("Kruti Dev legacy Hindi font detected. Auto-converting to Unicode Devanagari in 0.1s...")
                    pdf_text = kruti_to_unicode(pdf_text)
                elif is_garbled_text(pdf_text):
                    status.text("Legacy font / scanned PDF detected. Running Hindi OCR fallback...")
                    def update_ocr_status(current_page: int, total_pages: int) -> None:
                        progress_bar.progress(int(current_page / total_pages * 20))
                        status.text(f"Hindi OCR in progress: page {current_page}/{total_pages}. Please keep this tab open.")
                    pdf_text = ocr_hindi_pdf(uploaded_bytes, update_ocr_status)
                    if not pdf_text:
                        raise RuntimeError("Hindi OCR could not recover readable text from this PDF.")
                content = pdf_text.encode("utf-8")
            else:
                raw_text = uploaded_bytes.decode("utf-8", errors="ignore")
                if is_kruti_dev(raw_text):
                    status.text("Kruti Dev text detected. Auto-converting to Unicode Devanagari...")
                    raw_text = kruti_to_unicode(raw_text)
                    content = raw_text.encode("utf-8")
                else:
                    content = uploaded_bytes"""

code = code.replace(old_pdf_flow.replace('\r\n', '\n'), new_pdf_flow.replace('\r\n', '\n'))
code = code.replace(old_pdf_flow, new_pdf_flow)

# 4. In tab0: replace placeholder rating with real CBFC & IMDb Parents Guide panel
old_tab0_start = """            with tab0:
                st.subheader("Full screenplay analysis")
                rating_category = script_category(signal_score)
                summary_col, rating_col = st.columns([2.5, 1])
                with rating_col:
                    if unattributed_pdf:
                        st.markdown(\"\"\"
                        <div class="rating-panel">
                          <div class="rating-label">IMDb-style script category</div>
                          <div class="rating-value">N/A</div>
                          <div class="rating-note">This PDF has no speaker labels, so a character-aware rating cannot be calculated.</div>
                        </div>
                        \"\"\", unsafe_allow_html=True)
                    else:
                        st.markdown(f\"\"\"
                        <div class="rating-panel">
                          <div class="rating-label">IMDb-style script category</div>
                          <div class="rating-value">{rating_category}</div>
                          <div class="rating-note">Experimental dialogue-based category — not an official IMDb rating.</div>
                        </div>
                        \"\"\", unsafe_allow_html=True)
                with summary_col:
                    st.markdown("#### Story dialogue at a glance")
                    if unattributed_pdf:
                        st.write("The source PDF preserves dialogue wording but omits speaker names. You can inspect global language, emotion, and hesitation patterns, but character-level metrics cannot be trusted.")
                        st.caption("Current mode: unattributed dialogue analysis")
                    else:
                        st.write("This category is calculated from the full screenplay's dialogue variety, character participation, and turn-taking signals. It will become a real IMDb prediction only after training on real scripts and verified IMDb ratings.")
                        st.progress(int(max(0, min(100, signal_score))))
                        st.caption("Current mode: dialogue-based estimate")"""

new_tab0_start = """            with tab0:
                st.subheader("Screenplay Certification & Parents Guide")
                parents_guide = compute_parents_guide(df)
                cbfc_code = parents_guide['cbfc_code']
                mpaa_code = parents_guide['mpaa_code']
                advisory_summary = parents_guide['summary']

                cbfc_bg = "#d32f2f" if cbfc_code == "A" else ("#f57c00" if "16" in cbfc_code else ("#ffa000" if "13" in cbfc_code else ("#fbc02d" if "7" in cbfc_code else "#2e7d32")))
                mpaa_bg = "#d32f2f" if mpaa_code in ("R", "NC-17") else ("#f57c00" if mpaa_code == "PG-13" else ("#fbc02d" if mpaa_code == "PG" else "#2e7d32"))

                cert_col1, cert_col2 = st.columns([1.1, 2.9])
                with cert_col1:
                    st.markdown(f\"\"\"
                    <div style="background: white; border: 2px solid {cbfc_bg}; border-radius: 14px; padding: 1.1rem; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #555; text-transform: uppercase;">CBFC Certificate (India)</div>
                        <div style="display: inline-block; background: {cbfc_bg}; color: white; font-size: 2.2rem; font-weight: 850; padding: 0.2rem 1.4rem; border-radius: 10px; margin: 0.6rem 0;">{cbfc_code}</div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: #333;">{parents_guide['cbfc_rating']}</div>
                        <hr style="margin: 0.6rem 0; border: none; border-top: 1px dashed #ddd;">
                        <div style="font-size: 0.8rem; color: #666;">Global Equivalent: <span style="background: {mpaa_bg}; color: white; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{mpaa_code}</span> ({parents_guide['mpaa_rating']})</div>
                    </div>
                    \"\"\", unsafe_allow_html=True)
                with cert_col2:
                    st.markdown(f\"\"\"
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.1rem; height: 100%;">
                        <div style="font-size: 0.95rem; font-weight: 750; color: #1e293b; margin-bottom: 0.35rem;">IMDb Parents Guide & Content Advisory</div>
                        <div style="font-size: 0.85rem; color: #475569; line-height: 1.4; margin-bottom: 0.75rem;"><b>Reason for Rating:</b> {advisory_summary}</div>
                    </div>
                    \"\"\", unsafe_allow_html=True)
                    cat_cols = st.columns(5)
                    cat_names = [
                        ('violence', 'Violence & Gore'),
                        ('profanity', 'Profanity & Slurs'),
                        ('adult_themes', 'Sex & Nudity'),
                        ('substances', 'Alcohol & Drugs'),
                        ('psychological_intensity', 'Distress / Intensity')
                    ]
                    for idx, (cat_key, cat_title) in enumerate(cat_names):
                        cat_info = parents_guide['categories'][cat_key]
                        sev = cat_info['severity']
                        cnt = cat_info['count']
                        sev_color = "#d32f2f" if sev == "Severe" else ("#f57c00" if sev == "Moderate" else ("#fbc02d" if sev == "Mild" else "#2e7d32"))
                        with cat_cols[idx]:
                            st.markdown(f\"\"\"
                            <div style="background: white; border: 1px solid #cbd5e1; border-top: 4px solid {sev_color}; border-radius: 8px; padding: 0.5rem; text-align: center;">
                                <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; height: 1.8rem;">{cat_title}</div>
                                <div style="font-size: 0.92rem; font-weight: 750; color: {sev_color}; margin: 0.2rem 0;">{sev}</div>
                                <div style="font-size: 0.72rem; color: #94a3b8;">{cnt} flag(s)</div>
                            </div>
                            \"\"\", unsafe_allow_html=True)

                if parents_guide['character_breakdown']:
                    with st.expander("👤 Character Content Responsibility (Kaunsa character kitna offensive/violent hai)", expanded=False):
                        st.caption("Dialogues mein violence, slurs aur intense content use karne wale characters:")
                        char_df = pd.DataFrame(parents_guide['character_breakdown'])
                        st.dataframe(char_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Story Dialogue & Pacing Signals")"""

code = code.replace(old_tab0_start.replace('\r\n', '\n'), new_tab0_start.replace('\r\n', '\n'))
code = code.replace(old_tab0_start, new_tab0_start)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("app.py updated successfully!")
