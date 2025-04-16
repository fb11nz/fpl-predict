import os
import csv
import click

@click.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_csv", type=click.Path(writable=True))
@click.option("--dedupe", is_flag=True, help="Remove duplicate rows from the final output.")
def combine_filtered_csvs(input_folder, output_csv, dedupe):
    """Combine all .filtered.csv files into one CSV with only Bucket, Key, VersionId columns."""
    seen = set()
    total_written = 0

    with open(output_csv, "w", newline='') as out_file:
        writer = csv.writer(out_file)

        for filename in os.listdir(input_folder):
            if filename.endswith(".csv"):
                full_path = os.path.join(input_folder, filename)
                print(f"📄 Processing: {filename}")

                with open(full_path, "r", newline='') as in_file:
                    reader = csv.reader(in_file)

                    for row in reader:
                        if len(row) < 3:
                            print(f"⚠️ Skipping malformed row in {filename}: {row}")
                            continue

                        out_row = (row[0], row[1], row[2])  # Bucket, Key, VersionId

                        if dedupe and out_row in seen:
                            continue

                        writer.writerow(out_row)
                        total_written += 1

                        if dedupe:
                            seen.add(out_row)

    print(f"\n✅ Combined CSV written to: {output_csv}")
    print(f"🔢 Total rows written: {total_written}")

if __name__ == "__main__":
    combine_filtered_csvs()