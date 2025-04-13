import requests
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import find
import numpy as np

### 1. del: Iskanje publikacij, izvleček podatkov ############################
BASE_URL = "https://api.openalex.org/works"
SEARCH_QUERIES = [
    "artificial intelligence workplace surveillance",
    "artificial intelligence employee monitoring",
    "AI employee monitoring", "AI workplace surveillance",
    "digital workplace monitoring", "digital workplace ethics", 
    "algorithmic workplace surveillance",
    "digital worker surveillance", "remote work surveillance",
    "artificial intelligence employee tracking",
    "artificial intelligence workplace privacy",  
    "machine learning employee monitoring",      
    "AI workplace performance tracking",        
    "biometric workplace monitoring"             
]
YEAR_RANGE = "2015-2025"

# Iskanje publikacij
def fetch_papers(query, search_field="title.search"):
    papers = []
    for page in tqdm(range(1, 2), desc=f"'{query}' ({search_field})", leave=False):
        url = f"{BASE_URL}?filter={search_field}:{query},publication_year:{YEAR_RANGE}&per-page=200&page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            if not results:
                break
            papers.extend([p for p in results if p is not None and isinstance(p, dict)])
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching '{query}' ({search_field}) page {page}: {e}")
            break
    return papers

# Iz naslova članka izluščimo ključne besede, odstranimo nerelevantne
def get_keywords(title):
    stop_words = {"in", "of", "and", "the", "to", "from", "using" ,"is", "a", "on", "for", "with"}
    if title is None:
        return []
    words = title.lower().split()
    return [w for w in words if len(w) > 3 and w not in stop_words]

all_papers = []
seen_ids = set()
query_counts = {}

# Iskanje po seznamu ključnih besed
for query in tqdm(SEARCH_QUERIES, desc="Searching OpenAlex"):
    papers = fetch_papers(query, "title.search") + fetch_papers(query, "abstract.search")
    new_papers = [p for p in papers if p.get('id') and p['id'] not in seen_ids]
    query_counts[query] = len(new_papers)
    all_papers.extend(new_papers)
    seen_ids.update(p['id'] for p in new_papers)

# Izpis števila zadetkov po ključnih besedah
for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{query}: {count}")

# Ekstrakcija polj iz pridobljenih podatkov
data = []
for p in all_papers:
    try:
        data.append({
            "id": p.get("id", ""),
            "year": int(p.get("publication_year", 0) or 0),
            "title": p.get("title", ""),
            "authors": [
                a["author"].get("display_name", "")
                for a in p.get("authorships", [])
                if a.get("author")
            ],
            "institutions": [
                i.get("display_name", "")
                for a in p.get("authorships", [])
                for i in a.get("institutions", [])
                if i
            ],
            "journal": (
                p.get("primary_location", {}).get("source", {}).get("display_name", "")
                if p.get("primary_location")
                else ""
            ),
            "keywords": get_keywords(p.get("title", "")),
            "references": p.get("referenced_works", []),
        })
    except Exception as e:
        print(f"Skipping paper due to error: {e}")

df = pd.DataFrame(data)

# Filtriranje medicinskih in COVID-publikacij (velika količina poblikacij, ki niso relevantne za našo temo)
medicine_terms = [
    "medicine", "clinical", "hospital", "therapy", "treatment",
    "biomedicine", "pharmaceutical", "oncology", "cardiology", "malaria"
]
covid_terms = [
    "covid", "coronavirus", "sars-cov-2", "pandemic", "epidemic", "quarantine",
    "vaccine", "COVID-19"
]
exclude_terms = medicine_terms + covid_terms
exclude_journals = [
    "clinical", "virology", "epidemiology", "Diabetes"
]

def is_medical_or_covid(row):
    title = row["title"].lower() if row["title"] else ""
    keywords = [k.lower() for k in row["keywords"]] if row["keywords"] else []
    journal = row["journal"].lower() if row["journal"] else ""
    has_term = any(term in title or term in keywords for term in exclude_terms)
    has_medical_journal = any(jterm in journal for jterm in exclude_journals)
    return has_term or has_medical_journal

# Uporabi filter
df = df[~df.apply(is_medical_or_covid, axis=1)].reset_index(drop=True)
print(f"Filtriran podatkovni okvir: {len(df)} publikacij (odstranjene medicinske/COVID teme)")

df.to_csv("ai_surveillance_openalex.csv", index=False)


#### 2. del: Osnovna analiza ########################################
# Pregled podatkovnega okvirja
df.info()
df.head()
# Definiranje unikatnih polj
unique_works = df["title"].unique()
unique_authors = df["authors"].explode().unique() if "authors" in df.columns else []
unique_institutions = df["institution"].explode().unique() if "institution" in df.columns else []
unique_journals = df["journal"].unique() if "journal" in df.columns else []
unique_keywords = df["keywords"].explode().unique() if "keywords" in df.columns else []
# Print counts
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

# Najpogostejše revije
top_journals = df.explode("journal")["journal"].value_counts().head(10)
print("Top 10 revij:\n", top_journals)

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
print("Top 20 soavtorstev:")
for r, c, w in top_coauthors:
    print(f"{unique_authors[r]} — {unique_authors[c]}: {w} člankov")

# Citacije
# Dva pristopa, skupne citacije: identificira skupne citacije (sorodnost preko referenc).
# in Batageljev pristop izdelave matrik: sledi vplivu (kdo citira koga).

# Skupne Citacije
shared_citations = C @ C.T
shared_citations.setdiag(0)
rows, cols = shared_citations.nonzero()
weights = shared_citations.data
top_shared = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:10]
print("Top 10 skupnih citacij:")
for r, c, w in top_shared:
    print(f"{unique_works[r]} & {unique_works[c]} imata skupnih {w} citacij")
    
# Omrežje citiranj med avtorji po Batagelju (2008)
author_citation_matrix = A.T @ C @ A
author_citation_matrix.setdiag(0)
rows, cols = author_citation_matrix.nonzero()
weights = author_citation_matrix.data
top_author_cites = sorted(zip(rows, cols, weights), key=lambda x: -x[2])[:20]
print("Top 20 citiranj med avtorji:")
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
print("Top 20 so-pojavitev revij (prek avtorjev):")
for r, c, w in top_journal_cooccur:
    print(f"{unique_journals[r]} & {unique_journals[c]}: {w} skupnih avtorjev")
