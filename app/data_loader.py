import re
import os
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

def parse_pubmed_file_filtered(xml_path):
    articles = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for pubmed_article in root.findall(".//PubmedArticle"):
        article_data = {}

        pub_types = [
            pt.text.strip().lower()
            for pt in pubmed_article.findall(".//PublicationTypeList/PublicationType")
            if pt.text
        ]
        if any(pt in ["letter", "comment"] for pt in pub_types):
            continue

        abstract_elems = pubmed_article.findall(".//Abstract/AbstractText")
        if not abstract_elems:
            continue

        pmid_elem = pubmed_article.find(".//PMID")
        article_data["pmid"] = pmid_elem.text if pmid_elem is not None else None

        title_elem = pubmed_article.find(".//ArticleTitle")
        article_data["title"] = title_elem.text if title_elem is not None else None

        abstract_parts = []
        for elem in abstract_elems:
            label = elem.attrib.get("Label")
            text = elem.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        article_data["abstract"] = " ".join(abstract_parts)

        mesh_terms = [
            mesh.text
            for mesh in pubmed_article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if mesh.text
        ]
        article_data["mesh_terms"] = mesh_terms if mesh_terms else None

        pub_date_elem = pubmed_article.find(".//PubDate")
        year = None
        if pub_date_elem is not None:
            year_elem = pub_date_elem.find("Year")
            medline_date = pub_date_elem.find("MedlineDate")

            if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                year = int(year_elem.text)
            elif medline_date is not None and medline_date.text:
                match = re.search(r"\d{4}", medline_date.text)
                if match:
                    year = int(match.group())

        article_data["publication_year"] = year

        articles.append(article_data)

    return articles


def filter_pubmed_articles_by_topics(articles, topics, verbose=True):
    filtered = []
    for article in articles:
        text = (article["title"] or "") + " " + (article["abstract"] or "")
        mesh = article["mesh_terms"] or []
        if any(keyword.lower() in text.lower() for keyword in topics) or \
           any(any(keyword.lower() in term.lower() for keyword in topics) for term in mesh):
            filtered.append(article)

    if verbose:
        print(f"{len(filtered)} out of {len(articles)} PubMed articles matched topic filter")

    return filtered



def parse_pmc_file_filtered(xml_path, include_body=False):
    articles = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    if root.tag == "article":
        article = root
    else:
        article = root.find(".//article")
        if article is None:
            return []

    abstract_elem = article.find(".//abstract")
    if abstract_elem is None:
        return []

    abstract_text = " ".join(abstract_elem.itertext()).strip()
    if not abstract_text:
        return []

    pub_types = [
        pt.text.strip().lower()
        for pt in article.findall(".//publication-type")
        if pt.text
    ]
    if any(pt in ["letter", "comment"] for pt in pub_types):
        return []

    article_data = {}

    article_id = article.find(".//article-id[@pub-id-type='pmc']")
    article_data["pmcid"] = article_id.text if article_id is not None else None

    title_elem = article.find(".//title-group/article-title")
    article_data["title"] = title_elem.text.strip() if title_elem is not None and title_elem.text else None

    article_data["abstract"] = abstract_text

    year_elem = article.find(".//pub-date/year")
    article_data["publication_year"] = int(year_elem.text) if year_elem is not None and year_elem.text and year_elem.text.isdigit() else None

    keywords = [
        kw.text.strip()
        for kw in article.findall(".//kwd-group/kwd")
        if kw.text
    ]
    article_data["keywords"] = keywords if keywords else None

    if include_body:
        body_elem = article.find(".//body")
        if body_elem is not None:
            body_text = " ".join(body_elem.itertext()).strip()
            if body_text:
                article_data["body"] = body_text

    articles.append(article_data)
    return articles


def parse_folder_pmc(folder_path, include_body=False, limit=500):
    all_articles = []
    count = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue

        file_path = os.path.join(folder_path, filename)
        articles = parse_pmc_file_filtered(file_path, include_body=include_body)

        if articles:
            all_articles.extend(articles)
            count += 1
            if limit and count >= limit:
                break

    return all_articles


def filter_pmc_articles_by_topics(articles, topics, include_body_in_filter=True, verbose=True):
    filtered = []
    topics_lower = [t.lower() for t in topics]

    for article in articles:
        fields = [
            article.get("title", "") or "",
            article.get("abstract", "") or "",
            " ".join(article.get("keywords", []) or [])
        ]

        if include_body_in_filter:
            fields.append(article.get("body", "") or "")

        combined_text = " ".join(fields).lower()

        if any(topic in combined_text for topic in topics_lower):
            filtered.append(article)

    if verbose:
        print(f"{len(filtered)} out of {len(articles)} articles matched topic filter "
              f"(include_body_in_filter={include_body_in_filter})")

    return filtered


def prepare_pubmed_documents(articles):
    docs = []
    for article in articles:
        content = f"{article.get('title', '')}\n{article.get('abstract', '')}"
        metadata = {
            "source": "PubMed",
            "pmid": article.get("pmid", "unknown"),
            "title": article.get("title", "")
        }
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs



def prepare_pmc_documents(articles, chunk_size=1000, chunk_overlap=200):


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = []

    for article in articles:
        pmcid = article.get("pmcid", "unknown")
        title = article.get("title", "")
        abstract = article.get("abstract", "").strip()
        body = article.get("body", "").strip()

        if abstract:
            docs.append(Document(
                page_content=abstract,
                metadata={
                    "source": "PMC",
                    "pmcid": pmcid,
                    "title": title,
                    "chunk_id": -1,
                    "section": "abstract"
                }
            ))

        if body:
            chunks = splitter.split_text(body)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": "PMC",
                        "pmcid": pmcid,
                        "title": title,
                        "chunk_id": i,
                        "section": "body"
                    }
                ))

    return docs








