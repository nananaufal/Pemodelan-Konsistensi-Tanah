import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main(input_file, output_folder):
    # Membaca data dengan delimiter titik koma
    df = pd.read_csv(input_file, delimiter=';')

    # Cetak nama kolom dan beberapa baris pertama data
    print("Nama kolom:", df.columns)
    print(df.head())

    # Mengonversi kolom Konsistensi menjadi tipe kategori dengan urutan yang benar
    konsistensi_order = ['VERY SOFT', 'SOFT', 'MEDIUM', 'STIFF', 'VERY STIFF', 'HARD']
    df['Konsistensi'] = pd.Categorical(df['Konsistensi'], categories=konsistensi_order, ordered=True)

    # Membuat output folder jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Korelasi plot Kedalaman_qu dan Kedalaman_m
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Kedalaman_qu', y='Kedalaman_m', data=df)
    correlation = df['Kedalaman_qu'].corr(df['Kedalaman_m'])
    plt.title(f'Korelasi Kedalaman qu dan m: {correlation:.2f}')
    plt.savefig(os.path.join(output_folder, 'Korelasi_Kedalaman_qu_dan_m.png'))
    plt.close()

    # Boxplot Kedalaman_qu berdasarkan Konsistensi
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Konsistensi', y='Kedalaman_qu', data=df)
    plt.title('Boxplot Kedalaman_qu berdasarkan Konsistensi')
    plt.savefig(os.path.join(output_folder, 'Boxplot_Kedalaman_qu.png'))
    plt.close()

    # Boxplot Kedalaman_m berdasarkan Konsistensi
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Konsistensi', y='Kedalaman_m', data=df)
    plt.title('Boxplot Kedalaman_m berdasarkan Konsistensi')
    plt.savefig(os.path.join(output_folder, 'Boxplot_Kedalaman_m.png'))
    plt.close()

    # Plot Tabel Frekuensi Konsistensi dan Wilayah sebagai heatmap
    plt.figure(figsize=(12, 8))
    frequency_table = pd.crosstab(df['Wilayah'], df['Konsistensi'])
    sns.heatmap(frequency_table, annot=True, cmap='Blues', cbar=False)
    plt.title('Frequency Table Konsistensi dan Wilayah')
    plt.savefig(os.path.join(output_folder, 'Frequency_Table_Konsistensi_Wilayah.png'))
    plt.close()

    # Bar chart Jumlah Kategori Konsistensi
    plt.figure(figsize=(12, 8))
    konsistensi_counts = df['Konsistensi'].value_counts()
    sns.barplot(x=konsistensi_counts.index, y=konsistensi_counts.values)
    plt.title('Kategori Konsistensi')
    plt.xlabel('Konsistensi')
    plt.ylabel('Jumlah')
    plt.savefig(os.path.join(output_folder, 'Bar_Chart_Konsistensi.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')

    args = parser.parse_args()

    main(args.input, args.output)