# NLP Project: Song Lyrics Dataset

This project uses the [Genius Song Lyrics with Language Information](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) dataset from Kaggle for simple NLP experiments.

## Setup Instructions

### 1. Get Kaggle API Key
1. Go to your [Kaggle account settings](https://www.kaggle.com/).
2. Under the **API** section, click **Create New Token**.
3. This downloads a file called `kaggle.json`.

### 2. Place the API Key
Move the `kaggle.json` file into the **root of this project folder** (the same directory as your notebook).

Your project structure should look like this:

nlp-project/
│── kaggle.json
│── 01_import_data.ipynb
│── README.md


### 3. Run setup.ipynb to download files and libraries

## Data:

### Overview
- **Rows:** 5,134,856  
- **Columns:** 11  
- **Memory usage:** ~431 MB  

### Missing Values
| Column        | Missing |
|---------------|---------|
| title         | 188     |
| language_cld3 | 90,966  |
| language_ft   | 134,322 |
| language      | 226,918 |

---

### Key Columns

#### Title
- **Unique values:** ~3.1M  
- **Most common:** Intro (6,072), Home (1,826), Alone (1,617)  

#### Tag (Genre)
- **Categories:** pop, rap, rock, rb, misc, country  
- **Most common:** pop (2.1M), rap (1.7M), rock (0.8M)  

#### Artist
- **Unique values:** ~641K  
- **Most common sources:** Genius Romanizations (16K), Genius English Translations (13K)  

#### Year
- **Range:** 1 → 2100  
- **Median:** 2016  

#### Views
- **Range:** 0 → 23M  
- **Median:** 85  
- **Highly skewed distribution (a few very popular songs)**  

#### Language
- **Unique values:** 84  
- **Most common:** English (~3.3M), Spanish (~275K), French (~189K), Portuguese (~168K)  

---

### Notes
- The dataset is **large and sparse**, with many unique values (titles, artists, lyrics).  
- Genres (`tag`) are well-categorized and clean.  
- `views` is highly imbalanced, useful for popularity analysis.  
- `language` has multiple detection methods (`cld3`, `fastText`, manual), but with some missing data.  
