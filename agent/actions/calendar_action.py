from agent.actions.action import Action
from agent.controller.state import State
from tzlocal import get_localzone
from googleapiclient.discovery import build
import os.path
import pickle
from datetime import datetime, timedelta, timezone
import json
from agent.actions.utils import parse_payload, get_credentials

LOCAL_TIMEZONE = get_localzone()

class CalendarAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)

    def create_calendar_event(self, summary='Aura Slot', start_time=(datetime.now() + timedelta(minutes=10)).isoformat(), end_time=(datetime.now() + timedelta(minutes=25)).isoformat(), description=None):

        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time,
                'timeZone': str(LOCAL_TIMEZONE),
            },
            'end': {
                'dateTime': end_time,
                'timeZone': str(LOCAL_TIMEZONE),
            }
        }

        if description:
            event['description'] = description

        event = service.events().insert(calendarId='primary', body=event).execute()
        return event
    
    def delete_calendar_event(self, time: str = None):
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        dt = datetime.fromisoformat(time)
        dt = dt.replace(tzinfo=LOCAL_TIMEZONE)
        time_min = (dt - timedelta(days=1)).isoformat()
        time_max = (dt + timedelta(days=1)).isoformat()

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        for event in events_result.get('items', []):
            start_time = datetime.fromisoformat(event.get('start').get('dateTime'))
            end_time = datetime.fromisoformat(event.get('end').get('dateTime'))
            if dt >= start_time and dt <= end_time:
                service.events().delete(calendarId='primary', eventId=event['id']).execute()
                return f"Deleted event: {event.get('summary')}"
        
        return "No matching event found to delete"
        
    def is_valid_utc(self, dt_str):
        try:
            # Try to parse string in ISO 8601 UTC format
            datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
            return True
        except ValueError:
            print(f"Invalid UTC time: {dt_str}")
            return False

    def execute(self, state: State) -> str:
        info = parse_payload(self.payload)

        if 'start_time' in info and self.is_valid_utc(info['start_time']):
            start_time = info['start_time']
        else:
            start_time = datetime.now().isoformat()
        
        if 'end_time' in info and info['end_time'] is not None and self.is_valid_utc(info['end_time']):
            end_time = info['end_time']
        else:
            end_time = (datetime.fromisoformat(start_time) + timedelta(hours=1)).isoformat()
        
        if 'title' in info:
            summary = info['title']
        else:
            summary = 'Aura Slot'   
        
        if 'description' in info:
            description = info['description']
        else:
            description = 'This is a slot booked by Aura'
            
        try:
            if 'event'in info and info['event'] == 'delete':
                event = self.delete_calendar_event(start_time)
            else:
                event = self.create_calendar_event(
                    summary=summary,
                    start_time=start_time,
                    end_time=end_time,
                    description=description
                )
        except Exception as e:
            state.history.append({
                'action': {'type': 'calendar', 'payload': self.payload},
                'observation': f"Error creating calendar event: {e}"
            })
            print(f"Error creating calendar event: {e}")
            return f"Error creating calendar event: {e}"
        
        state.history.append({
            'action': {'type': 'calendar', 'payload': self.payload},
            'observation': f"Calendar event created"
        })

        return f"Calendar event created"

if __name__ == "__main__":
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    # calendar_list = service.calendarList().list().execute()

    # for calendar_list_entry in calendar_list['items']:
    #     print (calendar_list_entry)

    action = CalendarAction(thought="", payload=json.dumps({
        "event": "delete",
        "start_time": "2025-06-22T10:00:00",
        "title": "Train to London Kings Cross",
        "description": "I'll be leaving from Cambridge on the 22nd of April at 10am."
    }))
    action.execute(State())