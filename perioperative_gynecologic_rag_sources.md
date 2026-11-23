# Perioperative Gynecologic Nursing Care RAG Data Sources

This document lists recommended online sources and example approaches for downloading PDFs or retrieving metadata for a perioperative gynecologic nursing care RAG system.

> Notes:
> - Always review each website's Terms of Use and robots.txt before scraping.
> - Prefer official APIs when available.
> - Store metadata (source, publication year, topic, URL, trust level).
> - Re-check document freshness periodically.

---

# 1. NICE (National Institute for Health and Care Excellence)

## Main Site
https://www.nice.org.uk/guidance

## API
https://www.nice.org.uk/about/what-we-do/into-practice/nice-api

## Example Topics
- perioperative care
- surgical site infection
- venous thromboembolism
- hysterectomy
- postoperative pain management

## Example Search URL

```bash
https://api.nice.org.uk/services/search?q=perioperative%20care
```

## Example Python Download

```python
import requests

url = "https://www.nice.org.uk/guidance/ng180/resources/perioperative-care-pdf-123.pdf"
pdf = requests.get(url)

with open("nice_perioperative.pdf", "wb") as f:
    f.write(pdf.content)
```

---

# 2. RCOG (Royal College of Obstetricians and Gynaecologists)

## Main Site
https://www.rcog.org.uk/guidance/

## Suggested Topics
- hysterectomy
- enhanced recovery
- postoperative hemorrhage
- thromboprophylaxis

## Example Scraping Workflow

```python
import requests
from bs4 import BeautifulSoup

url = "https://www.rcog.org.uk/guidance/"
html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

pdf_links = []

for link in soup.find_all("a", href=True):
    href = link["href"]
    if ".pdf" in href:
        pdf_links.append(href)

print(pdf_links)
```

---

# 3. ACOG (American College of Obstetricians and Gynecologists)

## Main Site
https://www.acog.org/clinical

## Suggested Topics
- enhanced recovery after surgery
- hysterectomy
- laparoscopy
- postoperative urinary retention

## Example Search Query

```text
site:acog.org hysterectomy pdf
```

## Example Download Script

```python
import requests

pdf_url = "PDF_LINK_HERE"

r = requests.get(pdf_url)

with open("acog_guideline.pdf", "wb") as f:
    f.write(r.content)
```

---

# 4. AUGS (American Urogynecologic Society)

## Main Site
https://www.augs.org/clinical-patient-resources/clinical-guidance-documents/

## Suggested Topics
- catheter care
- postoperative bladder management
- urinary retention
- pelvic floor surgery

## Example PDF Discovery

```python
import requests
from bs4 import BeautifulSoup

url = "https://www.augs.org/clinical-patient-resources/clinical-guidance-documents/"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

for a in soup.find_all("a", href=True):
    if ".pdf" in a["href"]:
        print(a["href"])
```

---

# 5. CDC

## API Portal
https://developer.cdc.gov/apis

## Main Topics
- surgical site infection
- hand hygiene
- CAUTI prevention
- PPE

## Example Search

```text
site:cdc.gov surgical site infection pdf
```

---

# 6. WHO

## Main Site
https://www.who.int/

## API
https://www.who.int/data/gho/info/gho-odata-api

## Suggested Topics
- surgical safety checklist
- infection prevention
- maternal safety

## Example PDF Discovery

```python
import requests
from bs4 import BeautifulSoup

url = "https://www.who.int/publications"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

for a in soup.find_all("a", href=True):
    if ".pdf" in a["href"]:
        print(a["href"])
```

---

# 7. RNAO (Registered Nurses' Association of Ontario)

## Main Site
https://rnao.ca/bpg

## Recommended Topics
- pain management
- person-centered care
- transitions in care
- therapeutic communication

## Example Retrieval

```text
site:rnao.ca nursing best practice guideline pdf
```

---

# 8. AHRQ PSNet

## Main Site
https://psnet.ahrq.gov/

## Best Use
Reflective learning cases:
- near misses
- communication failures
- postoperative safety events

## Example Search

```text
postoperative gynecologic nursing site:psnet.ahrq.gov
```

---

# 9. PubMed

## API Documentation
https://www.ncbi.nlm.nih.gov/books/NBK25501/

## Example E-utilities Query

```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=perioperative+gynecologic+nursing
```

## Example Python

```python
import requests

url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

params = {
    "db": "pubmed",
    "term": "perioperative gynecologic nursing",
    "retmode": "json"
}

r = requests.get(url, params=params)

print(r.json())
```

---

# 10. Europe PMC

## API
https://europepmc.org/RestfulWebService

## Example Query

```bash
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=perioperative+gynecologic+nursing&format=json
```

---

# Recommended Initial Collection Strategy

## Phase 1 — High-Authority Guidelines
Collect:
- NICE
- RCOG
- ACOG
- CDC
- WHO

## Phase 2 — Nursing Practice Knowledge
Collect:
- RNAO
- OpenRN
- AUGS

## Phase 3 — Reflective Learning Cases
Collect:
- AHRQ PSNet
- public case studies
- communication-error examples

---

# Suggested Folder Structure

```text
data/
├── guidelines/
│   ├── NICE/
│   ├── RCOG/
│   ├── ACOG/
│   ├── CDC/
│   └── WHO/
├── nursing_practice/
│   ├── RNAO/
│   ├── OpenRN/
│   └── AUGS/
├── reflective_cases/
│   ├── PSNet/
│   └── case_examples/
└── metadata/
```

---

# Suggested Metadata Schema

```json
{
  "source": "RCOG",
  "document_type": "Guideline",
  "topic": "Postoperative Hemorrhage",
  "publication_year": 2025,
  "trust_level": 1,
  "url": "SOURCE_URL"
}
```

