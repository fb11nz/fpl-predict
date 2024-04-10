from datetime import datetime, timedelta
from pathlib import Path

from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

FILE_DIR = Path(__file__).resolve().parent

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

SHARED_CALENDAR_ID = 'c_133a42e880ad1b0e35aa2349459d973b7bbde92035b924300d575cab7dfdc665@group.calendar.google.com'


def get_calendar_events_for_today() -> list[dict]:
    credentials = ServiceAccountCredentials.from_json_keyfile_name(FILE_DIR / 'Employee status ZzShoe demo.json',
                                                                   scopes=SCOPES)

    service = build('calendar', 'v3', credentials=credentials)

    # Get events from the primary calendar
    today = datetime.now()
    offset_current = datetime.now().astimezone().utcoffset()
    today = today.replace(hour=0, minute=0, second=0, microsecond=0) - offset_current
    end_date = today + timedelta(days=1) - offset_current
    events_result = service.events().list(calendarId=SHARED_CALENDAR_ID, timeMin=today.isoformat() + 'Z',
                                          timeMax=end_date.isoformat() + 'Z', singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events


if __name__ == '__main__':
    all_events = get_calendar_events_for_today()
    for event in all_events:
        print(event['summary'])
