# Proyek Exploratory Data Analysis

Proyek ini bertujuan untuk melakukan analisis eksplorasi data pada dataset `data.csv` (pada project ini, folder data yang berisikan data.csv disembunyikan, jadi silahkan masukkan data.csv). Script `eda.py` akan menghasilkan plot dan tabel frekuensi yang disimpan di folder output yang Anda tentukan. Script `model.py` akan melakukan pemodelan dengan berbagai macam metode dan metadata yang berisi metrics akan di

## Langkah-langkah untuk Menjalankan Proyek

1. **Aktifkan virtual environment**:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate
    ```

2. **Install poetry untuk mengelola dependensi**:
    ```sh
    pip install poetry
    ```

3. **Instal semua dependensi yang diperlukan**:
    ```sh
    poetry install
    ```

4. **Jalankan script `eda.py` dan simpan visualisasi di folder yang diinginkan contohnya eda**:
    ```sh
    poetry run python eda.py --input data/data.csv --output eda
    ```

5. **Jalankan script `model.py`dan simpan metadata di folder yang diinginkan contohnya summary**:
    ```sh
    poetry run python model.py --input data/data.csv --output summary
    ``` 