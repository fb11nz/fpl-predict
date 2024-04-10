import csv

input_file = "Software.csv"
output_file = "software_ready.csv"


# Provide a mapping from input file column to output file column
# key = output file column name
# value = input file column name
column_header = [
    "system-name",
    "System administrator",
    "System URL",
    "System Overview",
    "Usage and Reasons for Adoption",
    "Start Date of Use",
    "Cloud or on-premise",
    "Location if on-premise",
    "holder of a privileged user ID",
    "Privileged ID shared or not",
    "If yes reason",
    "In-house users",
    "Availability of non-employee users",
    "If yes company or individual name",
    "Number of external users (accounts)",
    "Contract Plan",
    "Reasons for Plan Selection",
    "Authentication Method",
    "Usage Restrictions",
    "Logging availability",
    "Contents of logs to be obtained",
    "Handling of personal information of non-employees (customers applicants retirees etc.)",
    "Use of personal information other than system account information",
    "Information classification of the most important information to be stored",
    "Content of information to be stored selected in the information category",
    "Degree of impact in case of system outage",
    "update date",
]

matching_names = [
    "Software Name",
    "System Owner/Team",
    "Website",
    "Description",
    "Reason for Selection",
    "Date First Use",
    "Location",
    "",
    "Admin access",
    "Shared admin access",
    "Reason for shared access",
    "Users",
    "Available to contractors",
    "Name of contractors with access",
    "",
    "Type of plan",
    "Reason for plan selection",
    "Authentication Method",
    "Usage Restrictions",
    "Logging availability",
    "Contents of the logs",
    "Personal Data",
    "Type of personal data",
    "Classification",
    "Data Stored",
    "Impact of loss of service",
    "Date updated"
]

column_mapping = dict(zip(column_header, matching_names))

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
                column_index.append(header.index(name))
            except:
                print(name)

    # Fix bad handling of \n
    # loop each row
    all_rows = []
    for row in reader:
        if row[0] == "":
            for idx, val in enumerate(row):
                if val != "":
                    all_rows[-1][idx] = all_rows[-1][idx] + "\n" + val
            continue
        all_rows.append(row)



output_data = []
for row in all_rows:
    output_row = []

    # loop each column
    for dst_index, index in enumerate(column_index):
        # if column_header[dst_index] == "System administrator":
        #     output_row.append("Francis Bilham")

        if index == -1:
            output_row.append("-")    
        else:
            try:
                v = row[index]
                if column_header[dst_index] == "Start Date of Use": 
                    if v == "":
                        v = "01/04/2019"

                if v == "":
                    v = "-"
                elif column_header[dst_index] == "Degree of impact in case of system outage": 
                    if v == "Low":
                        v = "1 - Low"
                    elif v == "Medium":
                        v = "2 - Medium"
                    elif v == "High":
                        v = "3 - High"
                    else:
                        print("Bad format around '{}' -- {}".format(row[0], v))
                elif column_header[dst_index] == "Use of personal information other than system account information":
                    if v == "" or v == "-" or v == "None":
                        v = "3 - No personal information other than system account information is handled"
                    elif "payment" in v.lower() or \
                        "expenses" in v.lower() or \
                        "contracts" in v.lower() or \
                        "employee" in v.lower() or \
                        "personal" in v.lower() or \
                        "phone" in v.lower() or \
                        "login" in v.lower() or \
                        "visitor" in v.lower():
                        v = "2 - Obtain/store/pass on personal information other than system account information (includes handling of my number, sensitive information, and customer information)"
                    # elif v == "":
                    #     v = "1 - Obtain/store/pass on personal information other than system account information (does not include handling of my number, sensitive information, or customer information)""
                output_row.append(v)
            except IndexError:
                print("Bad format around '{}'".format(row[0]))
    output_data.append(output_row)
    # print(output_data)
    # break

# export output_data to csv, protect from comma in data
with open(output_file, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"')
    # writer.writerow(column_header)
    for row in output_data:
        writer.writerow(row)
