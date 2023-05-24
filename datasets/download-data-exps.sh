#!/bin/bash
function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

function github-get() {
    url=$1
    filename=$2
    output_dir=$3
    if [[ "${url}" == "" || "${filename}" == "" || "${output_dir}" == "" ]]; then
        echo "download_dataset url filename output_dir"
        return 1
    else
        echo "Downloading ${filename} from ${url} to ${output_dir}..."
        curl -LJO ${url}/${filename}
        mv ${filename} ${output_dir}/
        return 0
    fi
}

mkdir -p storage/

mkdir -p storage/datasets/electricity/
mkdir -p storage/datasets/traffic/
mkdir -p storage/datasets/wiki/



# Downloading determinist datasets
if [[ -f "storage/datasets/electricity/electricity.npy" ]]; then
    echo "File storage/datasets/electricity/electricity.npy already exists. Skipping download."
else
    gdrive-get 1UUwvY8Ixbwt3_fyDlJM80spZpgoOexRl storage/datasets/electricity/electricity.npy  # Download electricity dataset
fi

if [[ -f "storage/datasets/traffic/traffic.npy" ]]; then
    echo "File storage/datasets/traffic/traffic.npy already exists. Skipping download."
else
    gdrive-get 1dyeYj8IJwZ3bKvk1H67eaDTANdapKe7w storage/datasets/traffic/traffic.npy  # Download traffic dataset
fi

if [[ -f "storage/datasets/wiki/wiki.npy" ]]; then
    echo "File storage/datasets/wiki/wiki.npy already exists. Skipping download."
else
    gdrive-get 1VytXoL_vkrLqXxCR5IOXgE45hN2UL5oB storage/datasets/wiki/wiki.npy  # Download wiki dataset
fi
############################################



# Downloading already existing experiments
if [[ -d "storage/experiments" ]]; then
    echo "Directory storage/experiments already exists. Skipping download."
else
    gdrive-get 13xk12olmghlQLDOiGVL5gmFqDlKOfGaj storage/experiments.zip
    unzip storage/experiments.zip -d storage/
    rm storage/experiments.zip
fi
############################################



# Downloading probabilistic datasets
base_dir="storage/datasets/unextracted_ds"
base_github_url="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets"

mkdir -p ${base_dir}

if [[ -f "${base_dir}/electricity_nips.tar.gz" ]]; then
    echo "File ${base_dir}/electricity_nips.tar.gz already exists. Skipping download."
else
    github-get ${base_github_url} electricity_nips.tar.gz ${base_dir}  # Download electricity-small dataset
fi

if [[ -f "${base_dir}/solar_nips.tar.gz" ]]; then
    echo "File ${base_dir}/solar_nips.tar.gz already exists. Skipping download."
else
    github-get ${base_github_url} solar_nips.tar.gz ${base_dir}  # Download solar dataset
fi

if [[ -f "${base_dir}/taxi_30min.tar.gz" ]]; then
    echo "File ${base_dir}/taxi_30min.tar.gz already exists. Skipping download."
else
    github-get ${base_github_url} taxi_30min.tar.gz ${base_dir}  # Download taxi dataset
fi

if [[ -f "${base_dir}/wiki-rolling_nips.tar.gz" ]]; then
    echo "File ${base_dir}/wiki-rolling_nips.tar.gz already exists. Skipping download."
else
    github-get ${base_github_url} wiki-rolling_nips.tar.gz ${base_dir}  # Download wiki-small dataset
fi
############################################