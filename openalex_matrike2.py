import requests
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse import csr_matrix
from community import community_louvain
import numpy as np

sns.set_theme(style="whitegrid")
sns.set_palette("gray")

### 1. del: Iskanje publikacij, izvleček podatkov ############################
def fetch_papers(query, search_field="title.search"):
    papers = []
    for page in tqdm(range(1, 2), desc=f"'{query}' ({search_field})", leave=False):
        url = f"https://api.openalex.org/works?filter={search_field}:{query},publication_year:2015-2025&per-page=200&page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            if not results:
                break
            papers.extend([p for p in results if p is not None and isinstance(p, dict) and p.get('id')])
        except (requests.RequestException, ValueError) as e:
            print(f"Napaka pri pridobivanju '{query}' ({search_field}) stran {page}: {e}")
            break
    return papers

def get_keywords(title):
    if not title:
        return []
    stop_words = {"in", "of", "and", "the", "to", "from", "using", "is", "a", "on", "for", "with"}
    return [w for w in title.lower().split() if len(w) > 3 and w not in stop_words]

search_queries = [
    "artificial intelligence workplace surveillance", "artificial intelligence employee monitoring",
    "AI employee monitoring", "AI workplace surveillance", "digital workplace monitoring",
    "digital workplace ethics", "algorithmic workplace surveillance", "AI workplace ethics",
    "digital worker surveillance", "remote work surveillance", "artificial intelligence employee tracking",
    "artificial intelligence workplace privacy", "machine learning employee monitoring",
    "AI workplace performance tracking", "biometric workplace monitoring", "AI monitoring legislation",
    "AI GDPR surveillance", "EU AI Act workplace", "AI surveillance law"
]

all_papers = []
seen_ids = set()
query_counts = {}
for query in tqdm(search_queries, desc="Iskanje v OpenAlex"):
    papers = fetch_papers(query, "title.search") + fetch_papers(query, "abstract.search")
    new_papers = [p for p in papers if p.get('id') and p['id'] not in seen_ids]
    query_counts[query] = len(new_papers)
    all_papers.extend(new_papers)
    seen_ids.update(p['id'] for p in new_papers)

# Izpis števila zadetkov po ključnih besedah
for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{query}: {count}")

# Ustvari podatkovni okvir
data = []
for p in all_papers:
    try:
        if not p or not isinstance(p, dict):
            continue
        data.append({
            "id": p.get("id", ""),
            "year": int(p.get("publication_year", 0) or 0),
            "title": p.get("title", ""),
            "abstract_inverted_index": p.get("abstract_inverted_index", {}),
            "authors": [
                a["author"].get("display_name", "")
                for a in p.get("authorships", [])
                if a.get("author") and isinstance(a["author"], dict)
            ],
            "institutions": [
                i.get("display_name", "")
                for a in p.get("authorships", [])
                for i in a.get("institutions", [])
                if i and isinstance(i, dict)
            ],
            "journal": (
                p.get("primary_location", {}).get("source", {}).get("display_name", "")
                if p.get("primary_location") and isinstance(p.get("primary_location"), dict)
                else ""
            ),
            "keywords": get_keywords(p.get("title", "")),
            "references": p.get("referenced_works", [])
        })
    except Exception as e:
        print(f"Preskoči članek zaradi napake: {e}")

df = pd.DataFrame(data)
print(f"Začetni podatkovni okvir: {len(df)} člankov")

# Pretvori abstrakt v besedilo
def get_abstract_text(inverted_index):
    """Pretvori obrnjen indeks abstrakta v besedilo."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    max_pos = max([max(pos) for pos in inverted_index.values() if pos], default=0) + 1
    text = [""] * max_pos
    for word, positions in inverted_index.items():
        for pos in positions:
            text[pos] = word
    return " ".join([w for w in text if w])

df['abstract'] = df['abstract_inverted_index'].apply(get_abstract_text)

# Ekstrakcija polj
def get_main_field(concepts):
    if not concepts or not isinstance(concepts, list):
        return "Unknown"
    valid_concepts = [
        c for c in concepts
        if isinstance(c, dict) and 'display_name' in c and 'score' in c
    ]
    if not valid_concepts:
        return "Unknown"
    level1 = [c for c in valid_concepts if c.get('level') == 1]
    if level1:
        return max(level1, key=lambda c: float(c.get('score', 0)))['display_name']
    return max(valid_concepts, key=lambda c: float(c.get('score', 0)))['display_name']

def get_all_fields(concepts):
    if not concepts or not isinstance(concepts, list):
        return []
    valid_concepts = [
        c for c in concepts
        if isinstance(c, dict) and 'display_name' in c and 'score' in c
    ]
    if not valid_concepts:
        return []
    return [
        c['display_name']
        for c in sorted(valid_concepts, key=lambda c: float(c.get('score', 0)), reverse=True)
    ]

df['main_field'] = df['concepts'].apply(get_main_field)
df['fields'] = df['concepts'].apply(get_all_fields)

# Verifikacija
print("Top 10 main fields:")
print(df['main_field'].value_counts().head(10))
print("\nSample of all fields:")
print(df['fields'].head(5))

# Filtriranje medicinskih in COVID člankov
medicine_terms = ["medicine", "biomedicine", "pharmaceutical", "oncology", "cardiology", "malaria"]
covid_terms = ["covid", "coronavirus", "sars-cov-2", "pandemic", "epidemic", "quarantine", "vaccine", "COVID-19", "covid-19"]
exclude_terms = medicine_terms + covid_terms
exclude_journals = ["virology", "epidemiology", "Diabetes"]

def is_medical_or_covid(row):
    title = row["title"].lower() if row["title"] else ""
    keywords = [k.lower() for k in row["keywords"]] if row["keywords"] else []
    journal = row["journal"].lower() if row["journal"] else ""
    return (any(term in title or term in keywords for term in exclude_terms) or
            any(jterm in journal for jterm in exclude_journals))

df = df[~df.apply(is_medical_or_covid, axis=1)].reset_index(drop=True)

### 2. del: Osnovna analiza podatkov ###
# Pregled podatkovnega okvirja
df.info()
df.head()

# Pregled unikatnih polj
unique_works = df["title"].unique()
unique_authors = df["authors"].explode().unique() if "authors" in df.columns else []
unique_institutions = df["institution"].explode().unique() if "institution" in df.columns else []
unique_journals = df["journal"].unique() if "journal" in df.columns else []
unique_keywords = df["keywords"].explode().unique() if "keywords" in df.columns else []
print("Unikatna dela:", len(unique_works))
print("Unikatni avtorji:", len(unique_authors))
print("Unikatne inštitucije:", len(unique_institutions))
print("Unikatne revije:", len(unique_journals))
print("Unikatne ključne besede:", len(unique_keywords))

# Izpis prvih 15 del
print(df["title"].head(15).to_string(index=False))

# Najpogostejši avtorji
top_authors = df.explode("authors")["authors"].value_counts().head(10)
print("Top 10 avtorjev:\n", top_authors)

# Najpogostejše institucije
top_institutions = df.explode("institutions")["institutions"].value_counts().head(10)
print("Top 10 institucij:\n", top_institutions)

# Najpogostejše ključne besede
top_keywords = df.explode("keywords")["keywords"].value_counts().head(10)
print("Top 10 ključnih besed:\n", top_keywords)
plt.figure()
top_keywords.plot(kind="bar", title="Top 10 ključnih besed")
plt.xlabel("Ključna beseda")
plt.ylabel("Pojavitve")
plt.xticks(rotation=45, fontsize=9)
plt.tight_layout()
plt.show()

# Najpogostejše revije
top_journals = df['journal'].value_counts().head(50)
print("Top 50 Journals by Publication Count:")
print(top_journals)

# Pridobivanje področij za vsako revijo
journal_fields = {}
for journal in top_journals.index:
    journal_papers = df[df['journal'] == journal]
    # Združi vse 'fields' iz papirjev te revije
    all_fields = journal_papers['fields'].explode().dropna()
    # Vzemi najpogostejša področja (npr. top 5 po frekvenci)
    field_counts = all_fields.value_counts().head(5)
    journal_fields[journal] = field_counts.index.tolist()

# Analiza revij
print("\nFields of Top 50 Journals:")
for journal, fields in journal_fields.items():
    print(f"{journal}: {fields}")
all_journal_fields = pd.Series([field for fields in journal_fields.values() for field in fields])
field_counts = all_journal_fields.value_counts()
print("\nFrequency of Fields Across Top 50 Journals:")
print(field_counts.head(10))

### 3. del: Ključne besede skozi čas ###############################
df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])
keywords_df = df.explode("keywords")[["year", "keywords"]].dropna()
top_keywords = keywords_df["keywords"].value_counts().head(10).index
top_keywords_df = keywords_df[keywords_df["keywords"].isin(top_keywords)]
keyword_time = top_keywords_df.groupby(["year", "keywords"]).size().reset_index(name="count")
keyword_time["keywords"] = pd.Categorical(keyword_time["keywords"], categories=top_keywords, ordered=True)

# Grafi
sns.set_theme(style="whitegrid") 
sns.set_palette("gray")

plt.figure()
df["year"].value_counts().sort_index().plot(kind="bar", title="Število publikacij skozi čas", color="black")
plt.xlabel("Leto")
plt.ylabel("Število del")
plt.xticks(rotation=45, fontsize=9)
plt.tight_layout()
plt.show()

keywords_pivot = top_keywords_df.groupby(["year", "keywords"]).size().unstack(fill_value=0)[top_keywords]
keywords_pivot.plot(kind="line", figsize=(12, 6), title="Top 10 ključnih besed skozi čas")
plt.xlabel("Leto")
plt.ylabel("Frekvenca")
plt.legend(title="Ključne besede")
plt.tight_layout()
plt.show()

sns.set_theme(style="whitegrid")
sns.set_palette("gray")
g = sns.FacetGrid(keyword_time, col="keywords", col_wrap=2, sharey=False, height=3.5, aspect=1.4)
g.map_dataframe(sns.barplot, x="year", y="count", color="black")
g.set_titles("{col_name}", size=10)
g.set_axis_labels("Leto", "Frekvenca")
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=5)
plt.subplots_adjust(top=0.63, hspace=0.4)
g.fig.suptitle("Top 10 ključnih besed po letih", fontsize=14)
plt.tight_layout()
plt.show()

### 4. del: Ustvarjanje matrik #############################
unique_works = df["id"].unique()

# Dela × Avtorji (A)
works_authors = df[["id", "authors"]].explode("authors").dropna()
unique_authors = works_authors["authors"].unique().tolist()
A = csr_matrix(([1] * len(works_authors), (
    [list(unique_works).index(w) for w in works_authors["id"]],
    [list(unique_authors).index(a) for a in works_authors["authors"]]
)), shape=(len(unique_works), len(unique_authors)))

# Dela × Ključne besede (K)
works_keywords = df.explode("keywords")[["id", "keywords"]].dropna()
unique_keywords = works_keywords["keywords"].unique().tolist()
K = csr_matrix(([1] * len(works_keywords), (
    [list(unique_works).index(w) for w in works_keywords["id"]],
    [list(unique_keywords).index(k) for k in works_keywords["keywords"]]
)), shape=(len(unique_works), len(unique_keywords)))

# Dela × Leto (Y)
works_years = df[["id", "year"]].dropna()
unique_years = works_years["year"].unique()
Y = csr_matrix(([1] * len(works_years), (
    [list(unique_works).index(w) for w in works_years["id"]],
    [list(unique_years).index(y) for y in works_years["year"]]
)), shape=(len(unique_works), len(unique_years)))

# Dela × Revije (J)
works_journals = df[["id", "journal"]].dropna()
unique_journals = works_journals["journal"].unique().tolist()
J = csr_matrix(([1] * len(works_journals), (
    [list(unique_works).index(w) for w in works_journals["id"]],
    [list(unique_journals).index(j) for j in works_journals["journal"]]
)), shape=(len(unique_works), len(unique_journals)))

# Dela × Institucije (I)
works_institutions = df[["id", "institutions"]].explode("institutions").dropna()
unique_institutions = works_institutions["institutions"].unique().tolist()
I = csr_matrix(([1] * len(works_institutions), (
    [list(unique_works).index(w) for w in works_institutions["id"]],
    [list(unique_institutions).index(i) for i in works_institutions["institutions"]]
)), shape=(len(unique_works), len(unique_institutions)))

# Mreža citatov (C)
citations = df[["id", "references"]].explode("references").dropna()
citations = citations[citations["references"].isin(unique_works)]
C = csr_matrix(([1] * len(citations), (
    [list(unique_works).index(w) for w in citations["id"]],
    [list(unique_works).index(r) for r in citations["references"]]
)), shape=(len(unique_works), len(unique_works)))

### Del 5: Matrike Omrežij ########################################
# Soavtorstva
coauthor_matrix = A.T @ A
coauthor_matrix.setdiag(0)
rows, cols = coauthor_matrix.nonzero()
weights = coauthor_matrix.data
top_coauthors = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
for r, c, w in top_coauthors:
    print(f"{unique_authors[r]} — {unique_authors[c]}: {w} člankov")

# Dva pristopa, skupne citacije: identificira skupne citacije (sorodnost preko referenc).
# in Batageljev pristop izdelave matrik: sledi vplivu (kdo citira koga).

# Skupne Citacije
shared_citations = C @ C.T
shared_citations.setdiag(0)
rows, cols = shared_citations.nonzero()
weights = shared_citations.data
top_shared = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:10]
for r, c, w in top_shared:
    print(f"{unique_works[r]} & {unique_works[c]} imata skupnih {w} citacij")
    
# Omrežje citiranj med avtorji po Batagelju (2008)
author_citation_matrix = A.T @ C @ A
author_citation_matrix.setdiag(0)
rows, cols = author_citation_matrix.nonzero()
weights = author_citation_matrix.data
top_author_cites = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
for r, c, w in top_author_cites:
    print(f"{unique_authors[r]} je citiral {unique_authors[c]}: {w} krat")
# Batageljev model (2008) prikazuje tok citatov med avtorji — razkriva, kdo vpliva na koga in kateri avtorji so povezani prek citiranja.
# Dopolnjuje mrežo soavtorstev, saj prikazuje vpliv, ne le sodelovanje.

# Skupne revije (prek avtorjev)
journal_cooccur_matrix = J.T @ A @ A.T @ J
rows, cols, weights = find(journal_cooccur_matrix)
mask = rows != cols
rows, cols, weights = rows[mask], cols[mask], weights[mask]
top_journal_cooccur = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
for r, c, w in top_journal_cooccur:
    print(f"{unique_journals[r]} & {unique_journals[c]}: {w} skupnih avtorjev")

# Soavtorstvo institucij
inst_coauth_matrix = I.T @ I
inst_coauth_matrix.setdiag(0)
rows, cols = inst_coauth_matrix.nonzero()
weights = inst_coauth_matrix.data
top_inst_coauth = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
for r, c, w in top_inst_coauth:
    print(f"{unique_institutions[r]} & {unique_institutions[c]}: {w} člankov")

# Največja povezana komponenta - avtorji
G_authors = nx.Graph()
for r, c, w in zip(A.nonzero()[0], A.nonzero()[1], A.data):
    if r < c:
        G_authors.add_edge(unique_authors[r], unique_authors[c], weight=w)

components = list(nx.connected_components(G_authors))
largest_component = max(components, key=len)
largest_subgraph = G_authors.subgraph(largest_component)
print(f"Največja povezana komponenta: {len(largest_component)} avtorjev ({len(largest_component)/len(unique_authors)*100:.1f}% vseh)")
print(f"Povezave v NK: {largest_subgraph.number_of_edges()}")
top_authors_lcc = sorted([(a, largest_subgraph.degree(a)) for a in largest_component], key=lambda x: -x[1])[:20]
for a, deg in top_authors_lcc:
    print(f"{a}: {deg} soavtorjev")

# Louvain gručenje avtorjev
author_partition = community_louvain.best_partition(largest_subgraph)
author_clusters = pd.DataFrame({
    'author': list(largest_component),
    'cluster': [author_partition.get(a, -1) for a in largest_component]
})
print("\nSkupnosti avtorjev (Louvain):")
for cluster, authors in author_clusters.groupby('cluster')['author'].apply(list).items():
    print(f"Skupina {cluster}: {len(authors)} avtorjev")
    print(authors[:5], "...")

# Največja povezana komponenta - institucije
G_institutions = nx.Graph()
for r, c, w in zip(I.nonzero()[0], I.nonzero()[1], I.data):
    if r < c:
        G_institutions.add_edge(unique_institutions[r], unique_institutions[c], weight=w)

inst_components = list(nx.connected_components(G_institutions))
inst_largest_component = max(inst_components, key=len)
inst_largest_subgraph = G_institutions.subgraph(inst_largest_component)
print(f"Največja povezana komponenta: {len(inst_largest_component)} institucij ({len(inst_largest_component)/len(unique_institutions)*100:.1f}% vseh)")
print(f"Povezave v NK: {inst_largest_subgraph.number_of_edges()}")
top_insts_lcc = sorted([(i, inst_largest_subgraph.degree(i)) for i in inst_largest_component], key=lambda x: -x[1])[:10]
for i, deg in top_insts_lcc:
    print(f"{i}: {deg} povezav soavtorjev")

# Izvoz omrežja za Pajek
# Export the graphs using NetworkX's built-in formats
nx.write_graphml(G_keywords, "keyword_network.graphml")
nx.write_graphml(G_institutions, "institutions_network.graphml")
nx.write_graphml(A, "authors_network.graphml")
nx.write_graphml(J, "journals_network.graphml")
df.to_csv("ai_surveillance_openalex_updated.csv", index=False)

### 6. del: Tematska analiza ###
# Analiza sklopov ključnih besed
keyword_cooccur_matrix = K.T @ K
keyword_cooccur_matrix.setdiag(0)
rows, cols = keyword_cooccur_matrix.nonzero()
weights = keyword_cooccur_matrix.data
top_keyword_occ = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
for r, c, w in top_keyword_occ:
    print(f"{unique_keywords[r]} — {unique_keywords[c]}: {w} člankov")

G_keywords = nx.Graph()
for r, c, w in zip(rows, cols, weights):
    G_keywords.add_edge(unique_keywords[r], unique_keywords[c], weight=w)

partition = community_louvain.best_partition(G_keywords)
keyword_clusters = pd.DataFrame({
    'keyword': unique_keywords,
    'louvain_cluster': [partition.get(k, -1) for k in unique_keywords]
})
print(keyword_clusters.groupby('louvain_cluster')['keyword'].apply(list))

df['louvain_cluster'] = df['keywords'].apply(
    lambda ks: keyword_clusters[keyword_clusters['keyword'].isin(ks)]['louvain_cluster'].mode().get(0, -1)
)
print(df['louvain_cluster'].value_counts())

# Klasifikacija člankov
tech_terms = ["algorithm", "machine", "learning", "technology", "development", "application",
              "surveillance", "monitoring", "data", "system", "control"]
social_terms = ["ethics", "privacy", "wellbeing", "human", "social", "worker", "impact", "rights", "trust"]
legal_terms = ["gdpr", "regulation", "law", "legislation", "policy", "compliance", "act", "framework", "governance"]
medical_terms = ["biomedical", "clinical", "disease"]
economic_terms = ["market", "productivity", "economy", "cost", "labor", "financial", "policy", "governance"]
hr_terms = ["employee", "performance", "management", "organizational", "recruitment", "workplace", "leadership"]

def classify_paper(row):
    """Klasificira članek na podlagi ključnih besed."""
    keywords = set(row['keywords'])
    tech_score = len(keywords.intersection(tech_terms))
    social_score = len(keywords.intersection(social_terms))
    legal_score = len(keywords.intersection(legal_terms))
    medical_score = len(keywords.intersection(medical_terms))
    economic_score = len(keywords.intersection(economic_terms))
    hr_score = len(keywords.intersection(hr_terms))
    scores = {
        "Tehnološko": tech_score,
        "Socialno/Dobrobit": social_score,
        "Pravno": legal_score,
        "Medicinsko": medical_score,
        "Ekonomsko": economic_score,
        "Kadrovsko": hr_score
    }
    max_score = max(scores.values())
    return "Mešano" if max_score == 0 else max(scores, key=scores.get)

df['category'] = df.apply(classify_paper, axis=1)
print(df['category'].value_counts())

# Preverjanje omembe zakonodaje
legal_terms = ["gdpr", "GDPR", "AI Act", "regulation", "law", "legislation", "legal", "policy", "compliance"]
df['mentions_legislation'] = (
    df['abstract'].str.lower().str.contains('|'.join(legal_terms), na=False) |
    df['title'].str.lower().str.contains('|'.join(legal_terms), na=False)
)
print("\nČlanki z omembo zakonodaje:", df['mentions_legislation'].sum())
print(df[df['mentions_legislation']][['title', 'year']].head())

# Detekcija kritičnih člankov
critical_terms = ["ethics", "critical", "social impact", "privacy", "concerning"]
df['is_critical'] = (
    df['keywords'].apply(
        lambda ks: any(term in k.lower() for k in ks for term in critical_terms)
    ) |
    df['title'].str.lower().str.contains('|'.join(critical_terms), na=False)
)
print("\nKritični članki:", df['is_critical'].sum())
print(df[df['is_critical']][['title', 'year']].head())


# Preverjanje psiholoških študij
psych_terms = [
    "mental health", "anxiety", "depression", "burnout", "psychological wellbeing",
    "emotional distress", "job stress", "mental strain", "psychosocial", "employee wellbeing"
]
df['is_psych_study'] = (
    df['abstract'].str.lower().str.contains('|'.join(psych_terms), na=False) |
    df['title'].str.lower().str.contains('|'.join(psych_terms), na=False) |
    df['keywords'].apply(
        lambda ks: any(term in k.lower() for k in ks for term in psych_terms) if isinstance(ks, list) else False
    )
)
# Izpis rezultatov
print(f"\nŠtevilo psiholoških študij: {df['is_psych_study'].sum()} člankov")
print("Vzorec psiholoških študij:")
print(df[df['is_psych_study']][['title', 'year']].head(35))
