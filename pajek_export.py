import networkx as nx
import pandas as pd
from community import community_louvain
import scipy.sparse as sp
import numpy as np

# Load saved dataframe
df = pd.read_csv("ai_surveillance_openalex_updated.csv")
unique_authors = list(set([a for authors in df['authors'] for a in (eval(authors) if isinstance(authors, str) else [authors])]))
unique_institutions = list(set([i for insts in df['institutions'] for i in (eval(insts) if isinstance(insts, str) else [insts])]))
unique_keywords = list(set([k for keywords in df['keywords'] for k in (eval(keywords) if isinstance(keywords, str) else keywords)]))
unique_journals = df['journal'].dropna().unique()

# Function to write partitions
def write_partition(filename, nodes, partition):
    with open(filename, "w") as f:
        f.write(f"*Vertices {len(nodes)}\n")
        for node in nodes:
            f.write(f"{partition.get(node, -1)}\n")

# 1. Authors Network
G_authors = nx.Graph()
for authors in df['authors']:
    authors = eval(authors) if isinstance(authors, str) else [authors]
    for i, a1 in enumerate(authors):
        for a2 in authors[i+1:]:
            G_authors.add_edge(a1, a2, weight=G_authors.get_edge_data(a1, a2, default={'weight': 0})['weight'] + 1)
G_authors_filtered = G_authors.subgraph([n for n, d in G_authors.degree() if d >= 5]).copy()
nx.write_pajek(G_authors_filtered, "authors_network_filtered.net")
author_partition = community_louvain.best_partition(G_authors_filtered)
write_partition("authors_partition.clu", G_authors_filtered.nodes, author_partition)

# 2. Keywords Network
top_keywords = df.explode("keywords")["keywords"].value_counts().head(100).index
G_keywords = nx.Graph()
for keywords in df['keywords']:
    keywords = eval(keywords) if isinstance(keywords, str) else keywords
    for i, k1 in enumerate(keywords):
        for k2 in keywords[i+1:]:
            G_keywords.add_edge(k1, k2, weight=G_keywords.get_edge_data(k1, k2, default={'weight': 0})['weight'] + 1)
G_keywords_filtered = G_keywords.subgraph(top_keywords).copy()
G_keywords_filtered.remove_edges_from([(u, v) for u, v, d in G_keywords_filtered.edges(data=True) if d['weight'] < 10])
nx.write_pajek(G_keywords_filtered, "keywords_network_filtered.net")
keyword_partition = community_louvain.best_partition(G_keywords_filtered)
write_partition("keywords_partition.clu", G_keywords_filtered.nodes, keyword_partition)

# 3. Institutions Network
G_institutions = nx.Graph()
for insts in df['institutions']:
    insts = eval(insts) if isinstance(insts, str) else [insts]
    for i, i1 in enumerate(insts):
        for i2 in insts[i+1:]:
            G_institutions.add_edge(i1, i2, weight=G_institutions.get_edge_data(i1, i2, default={'weight': 0})['weight'] + 1)
G_institutions_largest = G_institutions.subgraph(max(nx.connected_components(G_institutions), key=len)).copy()
nx.write_pajek(G_institutions_largest, "institutions_network_largest.net")
inst_partition = community_louvain.best_partition(G_institutions_largest)
write_partition("institutions_partition.clu", G_institutions_largest.nodes, inst_partition)

# 4. Tripartite Network (Authors-Institutions-Keywords)
G_tripartite = nx.Graph()
for _, row in df.iterrows():
    authors = eval(row['authors']) if isinstance(row['authors'], str) else [row['authors']]
    insts = eval(row['institutions']) if isinstance(row['institutions'], str) else [row['institutions']]
    keywords = eval(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
    work_id = row['id']
    for a in authors:
        for i in insts:
            G_tripartite.add_edge(a, i, type="author-inst")
        G_tripartite.add_edge(a, work_id, type="author-paper")
    for k in keywords:
        G_tripartite.add_edge(work_id, k, type="paper-keyword")
degrees = dict(G_tripartite.degree())
top_nodes = (
    [n for n, _ in sorted([(n, d) for n, d in degrees.items() if n in unique_authors], key=lambda x: -x[1])[:100]] +
    [n for n, _ in sorted([(n, d) for n, d in degrees.items() if n in unique_institutions], key=lambda x: -x[1])[:50]] +
    [n for n, _ in sorted([(n, d) for n, d in degrees.items() if n in unique_keywords], key=lambda x: -x[1])[:100]] +
    [n for n, _ in sorted([(n, d) for n, d in degrees.items() if n in df['id'].values and d >= 2], key=lambda x: -x[1])[:100]]
)
G_tripartite_filtered = G_tripartite.subgraph(top_nodes).copy()
nx.write_pajek(G_tripartite_filtered, "tripartite_network.net")
tripartite_partition = community_louvain.best_partition(G_tripartite_filtered)
write_partition("tripartite_partition.clu", G_tripartite_filtered.nodes, tripartite_partition)

# 5. Union Network
G_union = nx.Graph()
for u, v, d in G_authors_filtered.edges(data=True):
    G_union.add_edge(u, v, weight=d['weight'], type="coauthorship")
for u, v, d in G_institutions_largest.edges(data=True):
    G_union.add_edge(u, v, weight=d['weight'], type="inst-collaboration")
for u, v, d in G_keywords_filtered.edges(data=True):
    G_union.add_edge(u, v, weight=d['weight'], type="keyword-cooccur")
for _, row in df.iterrows():
    authors = eval(row['authors']) if isinstance(row['authors'], str) else [row['authors']]
    insts = eval(row['institutions']) if isinstance(row['institutions'], str) else [row['institutions']]
    for a in authors:
        for i in insts:
            if a in G_authors_filtered.nodes and i in G_institutions_largest.nodes:
                G_union.add_edge(a, i, weight=1, type="author-inst")
nx.write_pajek(G_union, "union_network.net")
union_partition = {n: author_partition.get(n, inst_partition.get(n, keyword_partition.get(n, -1))) for n in G_union.nodes}
write_partition("union_partition.clu", G_union.nodes, union_partition)

# 6. Institution-Keyword Network
G_inst_keyword = nx.Graph()
inst_keyword_pairs = df.explode('institutions').explode('keywords')[['institutions', 'keywords']].dropna()
top_insts = [n for n, _ in sorted([(n, d) for n, d in G_institutions.degree() if n in unique_institutions], key=lambda x: -x[1])[:50]]
for inst1 in top_insts:
    keywords1 = set(inst_keyword_pairs[inst_keyword_pairs['institutions'] == inst1]['keywords'])
    for inst2 in top_insts:
        if inst1 < inst2:
            keywords2 = set(inst_keyword_pairs[inst_keyword_pairs['institutions'] == inst2]['keywords'])
            shared = len(keywords1.intersection(keywords2))
            if shared > 0:
                G_inst_keyword.add_edge(inst1, inst2, weight=shared)
nx.write_pajek(G_inst_keyword, "inst_keyword_network.net")
inst_keyword_partition = community_louvain.best_partition(G_inst_keyword)
write_partition("inst_keyword_partition.clu", G_inst_keyword.nodes, inst_keyword_partition)

# 7. Authors-Journals Network
G_authors_journals = nx.Graph()
for _, row in df.iterrows():
    authors = eval(row['authors']) if isinstance(row['authors'], str) else [row['authors']]
    journal = row['journal']
    for a in authors:
        G_authors_journals.add_edge(a, journal, weight=1)
top_nodes = (
    [n for n, _ in sorted([(n, d) for n, d in G_authors_journals.degree() if n in unique_authors], key=lambda x: -x[1])[:100]] +
    [n for n, _ in sorted([(n, d) for n, d in G_authors_journals.degree() if n in unique_journals], key=lambda x: -x[1])[:50]]
)
G_authors_journals_filtered = G_authors_journals.subgraph(top_nodes).copy()
nx.write_pajek(G_authors_journals_filtered, "authors_journals_network.net")
auth_journ_partition = community_louvain.best_partition(G_authors_journals_filtered)
write_partition("authors_journals_partition.clu", G_authors_journals_filtered.nodes, auth_journ_partition)

# 8. Article Classification
terms = {
    "Tehnološko": ["algorithm", "machine", "learning", "technology", "development", "application", "surveillance", "monitoring", "data", "system", "control"],
    "Socialno/Dobrobit": ["ethics", "privacy", "wellbeing", "human", "social", "worker", "impact", "rights", "trust"],
    "Pravno": ["gdpr", "regulation", "law", "legislation", "policy", "compliance", "act", "framework", "governance"],
    "Medicinsko": ["biomedical", "clinical", "disease"],
    "Ekonomsko": ["market", "productivity", "economy", "cost", "labor", "financial", "policy", "governance"],
    "Kadrovsko": ["employee", "performance", "management", "organizational", "recruitment", "workplace", "leadership"]
}
def classify_paper(keywords):
    if not isinstance(keywords, list):
        return "Mešano"
    scores = {cat: len(set(keywords).intersection(terms[cat])) for cat in terms}
    max_score = max(scores.values())
    return max(scores, key=scores.get) if max_score > 0 else "Mešano"
df['category'] = df['keywords'].apply(classify_paper)
category_to_int = {cat: i for i, cat in enumerate(set(df['category']))}
write_partition("article_categories.clu", df.index, {i: category_to_int[df.loc[i, 'category']] for i in df.index})

# Save updated dataframe
df.to_csv("ai_surveillance_openalex_updated.csv", index=False)
