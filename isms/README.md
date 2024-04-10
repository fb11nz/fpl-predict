# isms

## Export scripts

Both scripts have the same logic: take a spreadsheet as input and re-order the columns to match the desired output
`column_header` is nickname of the output columns (their actual string doesn't really matter)
`matching_names` is the name of the column to lookfor in the input spreadsheet. Leave empty if there is no matching column

There are also some overrides defined in the code, see for instance:
```
  if column_header[dst_index] == "Start Date of Use": 
    if v == "":
      v = "01/04/2019"
```
which will set a default date if the field is empty

### Export software list
1. Go to ISMS - Software Systems: https://zozonz.atlassian.net/wiki/spaces/ISMS/pages/2147221505/Software+Systems
2. Export page to Word document: ... > Export > Export to Word
3. Open word document, select first table, copy and paste it into Excel. The spreadsheet is a bit broken and needs to be fixed
4. With the whole spreadsheet still selected, click on "Unmerge cells" (within the Home tab). Some of the content is now broken into multiple cells. No worries, the script will handle it
5. Save spreadsheet in "CSV UTF-8 (Comma-delimited) (.csv)"
6. Open "converter-software.py" and change value of `input_file`
7. Run `$ python converter-software.py`
8. Open and check the output file in Excel: `software_ready.csv`
9. Copy/paste into ISMS document

### Export Private Data list
1. Go to JIRA - Information Asset filter: https://zozonz.atlassian.net/issues/?filter=10603
2. Export to csv: Export > Export CSV (all fields)"
3. Open "converter-private_data.py" and change value of `input_file`
4. Run `$ python converter-private_data.py`
5. Open and check the output file in Excel: `user_trial_ready.csv`
6. There is a bug between Excel and Google Spreadsheet: if a cell's content is overflowing and showing #######, the copy paste will fail!! You must first expand the column such that the content is visible
7. Copy/paste into ISMS document



