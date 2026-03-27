# Findings: Google and the Mind Experiment
Generated: 2026-03-25T18:17:47+00:00  
Corpus: Brown Corpus (fallback)  
Kenon version: 0.1.0

---

## 1. Headline result

**Best configuration:** `cooc_w2_backbone` + `betweenness`  
**Spearman r with norm PageRank:** 0.5391 (p=1.59e-297, n=3960 words)

The `cooc_w2_backbone` graph with `betweenness` centrality achieves the highest rank correlation with human free-association norms, explaining how word prominence in corpus-derived graphs reflects prominence in human associative memory.

---

## 2. Centrality measure rankings

Ranked by mean Spearman r across all graphs (mean ± std):

| Rank | Centrality measure | Mean r | Std | Best graph |
|---|---|---|---|---|
| 1 | betweenness | 0.1701 | 0.2617 | cooc_w2_backbone |
| 2 | degree | 0.1097 | 0.3044 | cooc_w2_backbone |
| 3 | pagerank | 0.0735 | 0.3413 | cooc_w2_backbone |

**Winner:** betweenness. **Interpretation:** This centrality measure most consistently recovers the relative prominence of words in human associative memory across different graph construction methods.

---

## 3. Graph method rankings

Ranked by best centrality r achieved:

| Rank | Graph | Best centrality | Best r | F1 | Backbone |
|---|---|---|---|---|---|
| 1 | cooc_w2_backbone | betweenness | 0.5391 | 0.0293 | yes |
| 2 | cooc_w2 | pagerank | 0.4746 | 0.0459 | no |
| 3 | count_backbone | betweenness | 0.0452 | 0.0159 | yes |
| 4 | count | betweenness | 0.0345 | 0.0271 | no |
| 5 | tfidf_backbone | betweenness | -0.0104 | 0.0223 | yes |
| 6 | tfidf | betweenness | -0.0558 | 0.0382 | no |

---

## 4. Window size effect (co-occurrence graphs)

For the best centrality measure (`pagerank`):

| Window | r (base) | r (backbone) | Δ |
|---|---|---|---|
| 2.0 | 0.4746 | 0.5331 | +0.0585 |

**Pattern:** insufficient data

---

## 5. Backbone effect

| Centrality | Mean Δr (backbone − base) | Consistent direction? |
|---|---|---|
| degree | +0.1055 | yes |
| pagerank | +0.1031 | yes |
| betweenness | +0.0425 | yes |

**Summary:** Backbone extraction on average improves centrality correlation with norms by 0.0837. 
This suggests that removing noisy edges with the disparity filter reveals structure more aligned with human associative memory.

---

## 6. Semantic vs. co-occurrence

| | Best co-occurrence | Best semantic |
|---|---|---|
| Graph | cooc_w2_backbone | count_backbone |
| Best centrality | betweenness | betweenness |
| Best r | 0.5391 | 0.0452 |
| F1 | 0.0293 | 0.0159 |

**Winner:** co-occurrence by 0.4939.

---

## 7. Recommended notebook configuration

```json
{
  "graph_type": "cooccurrence",
  "method": "cooc",
  "window": 2,
  "centrality_measure": "betweenness",
  "backbone": true,
  "spearman_r": 0.5391,
  "notebook_search_grid": {
    "window_range": [
      2,
      3
    ],
    "threshold_range": null,
    "backbone_alphas": [
      0.3,
      0.4,
      0.5,
      0.6,
      0.7
    ],
    "centralities_to_highlight": [
      "betweenness",
      "degree",
      "pagerank"
    ]
  }
}
```

---

## 8. Anomalies and caveats

- Negative correlations: count+degree, count+pagerank, count_backbone+pagerank, tfidf+degree, tfidf+pagerank

---

## 9. Raw results summary

| Graph | betweenness | degree | pagerank |
|---|---|---|---|
| cooc_w2 | 0.4678 | 0.4622 | 0.4746 |
| cooc_w2_backbone | 0.5391 | 0.5248 | 0.5331 |
| count | 0.0345 | -0.1301 | -0.1748 |
| count_backbone | 0.0452 | 0.0195 | -0.0169 |
| tfidf | -0.0558 | -0.1611 | -0.2338 |
| tfidf_backbone | -0.0104 | -0.0568 | -0.1410 |
