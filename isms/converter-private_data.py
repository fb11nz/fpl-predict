import csv

input_file = "Private.csv"
output_file = "user_trial_ready.csv"


# Provide a mapping from input file column to output file column
# key = output file column name
# value = input file column name
column_header = [
    "Administrator of the relevant personal information",
    "Members who can create edit reference and use this personal information",
    "Name of personal information",
    "Name of business handling personal information",
    "Services involved on the ZZJP side",
    "Type of Personal Information: employees or non-employees",
    "Information Category",
    "Personal Information Items",
    "Other items",
    "Number of cases of personal information handled",
    "Purpose of Use",
    "storage location",
    "Retention period / Utilization period",
    "Erase and Disposal Methods",
    "Date of acquisition",
    "source of acquisition",
    "Remarks on acquisition",
    "file format",
    "method of acquisition",
    "date of expiry",
    "update date",
]

matching_names = [
    "Assignee",
    "Access control",
    "Name of Asset",
    "Asset Created For Project",
    "ZOZO JP Service",
    "Afillition",
    "Asset Classification",
    "Type of data",
    "",
    "Number of records",
    "Purpose of using or holding data",
    "Location of Asset",
    "",
    "",
    "Asset Collected Start Date",
    "Source of data",
    "Comments",
    "File format",
    "Method of acquisition",
    "Access Expiry Date",
    "Updated",
]

column_mapping = dict(zip(column_header, matching_names))


output_data = []
with open(input_file) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # load header into dict
    header = next(reader)

    # find the column index for each matching_names
    column_index = []
    for name in matching_names:
        # Use index=-1 if name is empty
        if name == "":
            column_index.append(-1)
        else:
            try:
                idx = header.index(name)
            except ValueError:
                idx = header.index("Custom field ({})".format(name))
            column_index.append(idx)

    # loop each row
    for row in reader:
        output_row = []
        # loop each column
        for dst_index, index in enumerate(column_index):
            if column_header[dst_index] == "Retention period / Utilization period":
                output_row.append("End of business")
            elif column_header[dst_index] == "Erase and Disposal Methods":
                output_row.append("Delete data")

            elif index == -1:
                output_row.append("-")    
            else:
                try:
                    v = row[index]
                    if v == "":
                        v = "-"
                    output_row.append(v)
                except IndexError:
                    print("Bad format around '{}'".format(row[0]))
        output_data.append(output_row)
        # break

# export output_data to csv, protect from comma in data
with open(output_file, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"')
    # writer.writerow(column_header)
    for row in output_data:
        writer.writerow(row)
