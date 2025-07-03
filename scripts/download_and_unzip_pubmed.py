# PubMed & PMC Dataset Disclaimer:
# This project uses data from the U.S. National Library of Medicine (NLM),
# specifically the PubMed Baseline dataset and the PubMed Central (PMC) Open Access Subset.
# Both datasets are publicly available and may be freely used and redistributed under the terms below.
#
# Important Notes:
# - Use of this data does NOT imply endorsement by the NLM or the National Institutes of Health (NIH).
# - Do NOT use NIH or NLM logos, branding, or trademarks in your application or interface.
# - If your application modifies, summarizes, or generates outputs based on this data,
#   you must clearly state that the outputs are system-generated and NOT authored or approved by the NLM or NIH.
# - Articles in the PMC Open Access Subset are available under specific open-access licenses (e.g., CC BY, CC0),
#   and it is your responsibility to comply with the terms of each article’s license.
#
# License Terms:
# - PubMed Baseline: https://www.nlm.nih.gov/databases/download/terms_and_conditions.html
# - PMC OA License Info: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/


import re
import urllib.request
import gzip
import tarfile
import shutil
import os

from html.parser import HTMLParser


# Configuration
PUBMED_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PMC_OA_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"
DOWNLOAD_COUNT = 5  # Set to None to download all
OUTPUT_DIR = "data"

def download_file(url, output_path):
    print(f"\nDownloading: {url}")
    urllib.request.urlretrieve(url, output_path)

def extract_gzip(gz_path, out_path):
    print(f"Extracting GZip: {gz_path}")
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

def extract_tar(tar_path, extract_dir):
    print(f"Extracting TAR: {tar_path}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    os.remove(tar_path)

def download_pubmed_baseline(file_list_path):
    with open(file_list_path) as f:
        files = [line.strip() for line in f.readlines()]

    to_download = files if DOWNLOAD_COUNT is None else files[:DOWNLOAD_COUNT]
    print(f"\nDownloading {len(to_download)} PubMed baseline files...")
    for fname in to_download:
        gz_path = os.path.join(OUTPUT_DIR, fname)
        xml_path = os.path.join(OUTPUT_DIR, fname[:-3])
        download_file(PUBMED_BASE_URL + fname, gz_path)
        extract_gzip(gz_path, xml_path)


class TarballParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tarballs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.endswith(".tar.gz"):
                self.tarballs.append(href)

def download_pmc_oa_bulk():
    subdir = "oa_comm/xml/"
    index_url = PMC_OA_BASE_URL + subdir
    print(f"Fetching index from {index_url}")
    html = urllib.request.urlopen(index_url).read().decode()
    files = re.findall(r'href="([^"]+?\.tar\.gz)"', html)
    if not files:
        print("No .tar.gz files found in", subdir)
        return
    to_download = files if DOWNLOAD_COUNT is None else files[:DOWNLOAD_COUNT]
    for fname in to_download:
        url = index_url + fname
        download_file(url, os.path.join(OUTPUT_DIR, fname))
        extract_tar(os.path.join(OUTPUT_DIR, fname), OUTPUT_DIR)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== PubMed Baseline ===")
    download_pubmed_baseline("data/filelist.txt")

    print("\n=== PMC Open Access Subset ===")
    download_pmc_oa_bulk()

    print("\n✅ All done!")

if __name__ == "__main__":
    main()
