#! /usr/bin/env python3
import calendar
import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import gspread
import polars as pl
import requests
from dotenv import load_dotenv
from gspread import Cell
from gspread.worksheet import CellFormat
from oauth2client.service_account import ServiceAccountCredentials

import shared_calendar

FILE_DIR = Path(__file__).resolve().parent

load_dotenv()


@dataclass
class DeviceInfo:
    DeviceName: str
    InOffice: str


def filter_event_summaries_by_name(events: list[dict], name: str) -> list[str]:
    """
    Sorts through a list of dictionaries and returns a list of 'summary' items where 'name' is found in the summary.

    Returns:
        A list of 'summary' strings where the name is found in the summary.
    """

    filtered_summaries = []
    for item in events:
        summary = item.get('summary', '')  # Get summary, handle missing key
        if name.lower() in summary.lower():
            filtered_summaries.append(summary.lower())
    return filtered_summaries


def send_to_google_sheet(df_filtered):
    device_name_idx = df_filtered.columns.index("description")
    in_office_idx = df_filtered.columns.index("inOffice")
    devices_info = []
    for row in df_filtered.iter_rows():
        devices_info.append(
            DeviceInfo(
                DeviceName=row[device_name_idx],
                InOffice=row[in_office_idx]
            )
        )

    events_today = shared_calendar.get_calendar_events_for_today()
    # Authentication
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(FILE_DIR / 'Employee status ZzShoe demo.json', scope)
    client = gspread.authorize(credentials)

    # Open the Google Sheet

    # Get the current month name
    today = datetime.date.today()
    month_name = today.strftime("%B")
    year_name = today.strftime("%Y")
    sheet_name = f'{year_name} {month_name}'

    # Open the sheet or create it if it doesn't exist
    sheet = client.open_by_key('1RrY0qPg9hI747OE7oAZIbIrGvc6GBduSdFTZ_KwKYV4')
    person_device_sheet = sheet.worksheet("PersonDeviceLookup")
    person_lookup_data = {
        row[1]: row[0] for row in person_device_sheet.get_all_values()[1:]
    }

    # add missing devices
    all_devices = [x.DeviceName for x in devices_info]
    for device, person_name in person_lookup_data.items():
        if device not in all_devices:
            devices_info.append(
                DeviceInfo(
                    DeviceName=device,
                    InOffice='Offline'
                )
            )

    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows=100, cols=32)

    # Prepare headers (dates)
    _, num_days = calendar.monthrange(today.year, today.month)
    header_dates = [
        f"{today.month}/{day + 1} ({calendar.day_name[datetime.date(today.year, today.month, day + 1).weekday()]})"
        for day in range(num_days)
    ]
    header = ['Person', 'Device'] + header_dates

    all_cells = []
    all_cell_formats = []
    for col, header_item in enumerate(header):
        cell = Cell(row=1, col=col + 1, value=header_item)
        all_cells.append(cell)

    # Prepare device names
    existing_devices = worksheet.col_values(2)[1:]  # Fetch excluding header
    all_devices = [x.DeviceName for x in devices_info]
    new_devices = sorted([dev for dev in all_devices if dev not in existing_devices])
    if new_devices:  # Update only if there are new devices
        next_empty_row = len(existing_devices) + 2  # Start after existing data
        for new_dev in new_devices:
            cell = Cell(row=next_empty_row, col=2, value=new_dev)
            all_cells.append(cell)
            next_empty_row += 1
    devices = existing_devices + new_devices

    day_of_month = today.day
    target_column = day_of_month + 2  # Adjust if your columns start from a different index

    # Extract descriptions and update cells
    for i, device in enumerate(devices_info):
        row = devices.index(device.DeviceName) + 2
        col = target_column

        # Set person_name
        person_name = person_lookup_data.get(device.DeviceName, '')
        cell = Cell(row=row, col=1, value=person_name)
        all_cells.append(cell)

        cell = Cell(row=row, col=col, value="")
        all_cells.append(cell)
        # find all events with persons name in the summary

        if person_name:
            event_summaries = filter_event_summaries_by_name(events_today, person_name)
        else:
            event_summaries = []
        public_holiday = len(filter_event_summaries_by_name(events_today, "public holiday")) > 0
        if device.InOffice == "Online":
            cell_format = CellFormat(range=cell.address, format={"backgroundColor": {"red": 0, "green": 1, "blue": 0}})
        elif public_holiday:
            cell_format = CellFormat(range=cell.address,
                                     format={"backgroundColor": {"red": 1, "green": 0.8, "blue": 0.6}})
        elif any('leave' in x for x in event_summaries):
            cell_format = CellFormat(range=cell.address,
                                     format={"backgroundColor": {"red": 0.8, "green": 0.6, "blue": 0.8}})
        elif any('sick' in x for x in event_summaries):
            cell_format = CellFormat(range=cell.address,
                                     format={"backgroundColor": {"red": 1, "green": 0.75, "blue": 0.2}})
        else:
            cell_format = CellFormat(range=cell.address, format={"backgroundColor": {"red": 1, "green": 1, "blue": 1}})
        all_cell_formats.append(cell_format)
    worksheet.update_cells(all_cells)
    worksheet.batch_format(all_cell_formats)

    # Calculate for current status
    current_status_sheet = sheet.worksheet("CurrentStatus")
    header = ["Person", "Device", "Status", f"Updated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    all_cells = []
    all_cell_formats = []
    for col, header_item in enumerate(header):
        cell = Cell(row=1, col=col + 1, value=header_item)
        all_cells.append(cell)

    device_name_idx = df_filtered.columns.index("description")
    in_office_idx = df_filtered.columns.index("inOffice")
    devices = sorted(all_devices)
    for i, df_row in enumerate(df_filtered.iter_rows()):
        device_name = df_row[device_name_idx]
        row = devices.index(device_name) + 2
        cell = Cell(row=row, col=2, value=device_name)
        all_cells.append(cell)

        person_name = person_lookup_data.get(device_name, '')
        cell = Cell(row=row, col=1, value=person_name)
        all_cells.append(cell)

        in_office = df_row[in_office_idx]
        cell = Cell(row=row, col=3, value=in_office)
        all_cells.append(cell)
        if in_office == "Online":
            cell_format = CellFormat(range=cell.address, format={"backgroundColor": {"red": 0, "green": 1, "blue": 0}})
        else:
            cell_format = CellFormat(range=cell.address, format={"backgroundColor": {"red": 1, "green": 1, "blue": 1}})
        all_cell_formats.append(cell_format)

    current_status_sheet.update_cells(all_cells)
    current_status_sheet.batch_format(all_cell_formats)


def main():
    api_key = os.getenv('MERAKI_API_KEY')
    org_id = "601793500207411081"
    network_id = "L_601793500207435626"
    headers = {
        "X-Cisco-Meraki-API-Key": api_key
    }

    url = f"https://api.meraki.com/api/v1/networks/{network_id}/clients"
    response = requests.get(url, headers=headers, params={
        'perPage': 5000,

    })

    if response.status_code != 200:
        raise RuntimeError(f"HTTP status code {response.status_code}: {response.text}")
    devices = response.json()
    df = pl.DataFrame(devices)
    windows_devices = pl.col("os").str.contains("Windows")
    mac_devices = pl.col("deviceTypePrediction").str.contains("MacBook")
    combined_filter = windows_devices | mac_devices
    df_filtered = df.filter(combined_filter)
    today = datetime.date.today()
    df_filtered = df_filtered.with_columns(
        pl.col('lastSeen').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").dt.convert_time_zone(
            "Pacific/Auckland").dt.date().alias('parsed_datetime')
    )
    df_filtered = df_filtered.sort('lastSeen', descending=True)
    df_filtered = df_filtered.unique(subset='description', keep='first')

    df_filtered = df_filtered.with_columns(
        pl.when(pl.col('parsed_datetime').dt.date() == today)
        .then(pl.lit("Online"))
        .otherwise(pl.lit("Offline"))
        .alias("inOffice")
    )
    send_to_google_sheet(df_filtered)


if __name__ == "__main__":
    print(f"Running update at {datetime.datetime.now()}")
    main()
    print(f"Finished updating at {datetime.datetime.now()}")
