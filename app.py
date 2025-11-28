import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matching import (
    compute_similarity_matrix,
    greedy_weighted_matching,
    hopcroft_karp_matching,
    hungarian_matching,
)


st.set_page_config(page_title='Student–Internship Matcher', layout='wide')
st.title('Student Skill Graph — Internship Matching System')


st.sidebar.header('Upload your data')
students_file = st.sidebar.file_uploader('Upload Students CSV', type=['csv'])
companies_file = st.sidebar.file_uploader('Upload Companies CSV', type=['csv'])
algo = st.sidebar.selectbox(
    'Select Matching Algorithm',
    ['Greedy (weighted)', 'Hopcroft–Karp (max cardinality)', 'Hungarian (max weight)']
)
threshold = st.sidebar.slider(
    'Similarity threshold (edge exists if sim ≥ threshold)',
    0.0, 1.0, 0.0, 0.05
)

if students_file and companies_file:
    
    try:
        students = pd.read_csv(students_file, quotechar='"', engine='python')
        companies = pd.read_csv(companies_file, quotechar='"', engine='python')
    except Exception:
        students = pd.read_csv(students_file, sep=None, engine='python')
        companies = pd.read_csv(companies_file, sep=None, engine='python')

    
    if 'positions' not in companies.columns:
        st.error("The companies CSV must include a 'positions' column (number of openings per company).")
        st.stop()

    st.header('Input Preview')
    st.write(f"**Total Students:** {len(students)} | **Total Companies:** {len(companies)}")

    st.subheader('Students')
    st.dataframe(students, use_container_width=True, height=400)

    st.subheader('Companies (with positions)')
    st.dataframe(companies, use_container_width=True, height=400)

   
    expanded_companies = []
    for idx, row in companies.iterrows():
        for p in range(int(row['positions'])):
            new_row = row.copy()
            new_row['name'] = f"{row['name']} (Pos {p+1})"
            expanded_companies.append(new_row)
    expanded_companies = pd.DataFrame(expanded_companies).reset_index(drop=True)

    
    st.header('Skill Similarity Matrix')
    sim = compute_similarity_matrix(students, expanded_companies, skills_col='skills')
    sim_display = sim.copy()
    sim_display.index = students['name'].values
    sim_display.columns = expanded_companies['name'].values
    st.dataframe(sim_display.round(3), use_container_width=True, height=400)
    st.write("Similarity value range:",
         f"Min = {sim.values.min():.3f}, Max = {sim.values.max():.3f}")

    
    if st.button('Run Matching'):
        if algo == 'Greedy (weighted)':
            matches = greedy_weighted_matching(sim, threshold=threshold)
        elif algo == 'Hopcroft–Karp (max cardinality)':
            matches = hopcroft_karp_matching(sim, threshold=threshold)
        elif algo == 'Hungarian (max weight)':
            matches = hungarian_matching(sim)

        rows = []
        for s_idx, c_idx, score in matches:
            rows.append({
                'student_idx': int(s_idx),
                'student': students.loc[int(s_idx), 'name'],
                'company_idx': int(c_idx),
                'company': expanded_companies.loc[int(c_idx), 'name'],
                'original_company': expanded_companies.loc[int(c_idx), 'name'].split(' (Pos')[0],
                'similarity_score': round(float(score), 3)
            })

        st.divider()
        st.subheader('Matching Results')
        st.write(f"Total Students: {len(students)}")
        st.write(f"Total Companies: {len(companies)} (Total Positions: {len(expanded_companies)})")
        st.write(f"Matching Found: {len(rows)}")

        if rows:
            df_matches = pd.DataFrame(rows)
            st.dataframe(df_matches, use_container_width=True, height=400)

            # ✅ Download button
            csv = df_matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                'Download Matching CSV',
                csv,
                file_name='matching.csv',
                key='download_matches'
            )

           
            st.subheader("Bipartite Graph: Students ↔ Company Positions")

            G = nx.Graph()
            student_nodes = [f"S_{i}" for i in range(len(students))]
            company_nodes = [f"C_{j}" for j in range(len(expanded_companies))]
            G.add_nodes_from(student_nodes, bipartite=0)
            G.add_nodes_from(company_nodes, bipartite=1)

            for s_idx, c_idx, score in matches:
                G.add_edge(f"S_{s_idx}", f"C_{c_idx}", weight=score)

            total_students = len(students)
            total_companies = len(expanded_companies)
            max_nodes = max(total_students, total_companies)

            vertical_spacing = 1.2
            student_x, company_x = 0, 3

            pos = {}
            for i, node in enumerate(student_nodes):
                pos[node] = (student_x, i * (max_nodes / total_students) * vertical_spacing)
            for i, node in enumerate(company_nodes):
                pos[node] = (company_x, i * (max_nodes / total_companies) * vertical_spacing)

            edge_x, edge_y = [], []
            for s_idx, c_idx, score in matches:
                s_node = f"S_{s_idx}"
                c_node = f"C_{c_idx}"
                x0, y0 = pos[s_node]
                x1, y1 = pos[c_node]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color="rgba(0, 160, 0, 0.5)"),
                hoverinfo='none',
                mode='lines'
            )

            student_trace = go.Scatter(
                x=[pos[n][0] for n in student_nodes],
                y=[pos[n][1] for n in student_nodes],
                mode='markers+text',
                text=[students.loc[int(n.split('_')[1]), 'name'] for n in student_nodes],
                textposition="middle right",
                textfont=dict(size=10, color='black'),  
                marker=dict(size=8, color='blue', opacity=0.8),
                name='Students'
            )

            company_trace = go.Scatter(
                x=[pos[n][0] for n in company_nodes],
                y=[pos[n][1] for n in company_nodes],
                mode='markers+text',
                text=[expanded_companies.loc[int(n.split('_')[1]), 'name'] for n in company_nodes],
                textposition="middle left",
                textfont=dict(size=10, color='black'), 
                marker=dict(size=9, color='red', opacity=0.9),
                name='Companies'
            )

            
            fig = go.Figure(data=[edge_trace, student_trace, company_trace])
            fig.update_layout(
                showlegend=True,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                height=min(4000, int(35 * max_nodes)),
                title=f"Bipartite Graph — {len(students)} Students vs {len(expanded_companies)} Positions",
                plot_bgcolor='white',   
                paper_bgcolor='white',  
                font=dict(color='black')  
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info('No matches found for current threshold or algorithm.')
else:
    st.info('Please upload both students.csv and companies.csv to start.')
