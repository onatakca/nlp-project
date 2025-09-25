# Dataset Exploration Report (NLP Focus)

## Overview
- **Rows:** 5,134,856  
- **Columns:** 11  
- **Main features of interest:**  
  - `lyrics` (text data for NLP)  
  - `tag` (genre label for classification)  

## Target Variable: `tag`
- **Genres:** pop, rap, rock, rb, misc, country  
- **Distribution:**  
  - pop → 2.1M songs  
  - rap → 1.7M songs  
  - rock → 0.8M songs  
  - rb → 196K songs  
  - misc → 181K songs  
  - country → 100K songs  
- **Note:** The dataset is **imbalanced** — pop/rap dominate.

## Input Variable: `lyrics`
- **Unique lyrics entries:** ~5.06M  
- **Data quality issues:**  
  - Some placeholder text (e.g., “Tell us that you would like to have the lyrics of this song…”)  
  - Repeated entries for translations or missing lyrics.  
- **Most common “lyrics” are not actual songs**, so preprocessing will be required.  

## Notes for Next Steps
- Clean `lyrics` column (remove boilerplate / placeholder text).  
- Handle **class imbalance** in `tag`.  
- Explore NLP pipelines (tokenization, embeddings, classification models).  
- Consider filtering out rows with missing or placeholder lyrics to improve model quality.  
