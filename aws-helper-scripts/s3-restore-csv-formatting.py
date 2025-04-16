import os
import csv
import click

# Expected schema
expected_headers = [
    "Bucket", "Key", "VersionId", "IsLatest",
    "IsDeleteMarker", "StorageClass", "IntelligentTieringAccessTier"
]

def should_keep(row):
    # Keep only latest versions in INTELLIGENT_TIERING that are in archive tiers
    if row["IsLatest"].lower() != "true":
        return False
    if row["StorageClass"].strip().upper() != "INTELLIGENT_TIERING":
        return False
    if row["IntelligentTieringAccessTier"].strip().upper() not in {"ARCHIVE", "DEEP_ARCHIVE"}:
        return False
    return True

@click.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path(file_okay=False))
def filter_inventory(input_folder, output_folder):
    """Filter S3 inventory CSV files from INPUT_FOLDER and write filtered files to OUTPUT_FOLDER."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", newline='') as infile, open(output_path, "w", newline='') as outfile:
                reader = csv.DictReader(infile, fieldnames=expected_headers)
                writer = csv.DictWriter(outfile, fieldnames=expected_headers)

                for row in reader:
                    if should_keep(row):
                        writer.writerow(row)

    click.echo(f"✅ Filtering complete. Filtered files saved to: {output_folder}")

if __name__ == "__main__":
    filter_inventory()