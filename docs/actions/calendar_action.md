# Calendar Action

The Calendar Action is used to manage calendar events. It can create, read, update, and delete events on the user's calendar.

## Overview

The Calendar Action is implemented in the `CalendarAction` class, which extends the `Action` class. It is designed to interact with calendar services to manage events.

## Capabilities

The Calendar Action can:

- Create new calendar events
- Delete existing calendar events
- Retrieve calendar events
- Update calendar events

## Implementation

The Calendar Action is implemented in the `agent/actions/calendar_action.py` file. It uses the following components:

- **Action Base Class**: Extends the `Action` class defined in `agent/actions/action.py`.
- **Calendar API Integration**: Uses external calendar APIs to manage events.
- **Result Processing**: Processes and formats calendar operation results for the agent.

## Usage

The Calendar Action is used by the agent to manage calendar events. To use the Calendar Action:

1. Create a new instance of the `CalendarAction` class with the appropriate thought and payload:
   ```python
   from agent.actions.calendar_action import CalendarAction

   action = CalendarAction(
       thought="I should create a calendar event for the meeting",
       payload={
           "event": "create",
           "start_time": "2025-07-10T14:00:00",
           "end_time": "2025-07-10T15:00:00",
           "title": "Team Meeting",
           "description": "Weekly team sync-up meeting"
       }
   )
   ```

2. Execute the action with the current state:
   ```python
   observation = action.execute(state)
   ```

3. The observation will contain the result of the calendar operation.

## Example

Here's an example of how the Calendar Action is used to create a calendar event:

1. Agent creates a Calendar Action:
   ```python
   action = CalendarAction(
       thought="I should create a calendar event for the doctor's appointment",
       payload={
           "event": "create",
           "start_time": "2025-07-15T10:00:00",
           "end_time": "2025-07-15T11:00:00",
           "title": "Doctor's Appointment",
           "description": "Annual check-up with Dr. Smith"
       }
   )
   ```

2. Agent executes the action:
   ```python
   observation = action.execute(state)
   ```

3. The action creates a calendar event for the doctor's appointment on July 15, 2025, from 10:00 AM to 11:00 AM.

4. The observation contains the result of the calendar operation, which might include:
   - Confirmation that the event was created
   - Details of the created event
   - Any errors or warnings that occurred during the operation

5. The agent processes the observation and creates a new action based on the result, typically a Chat Action to confirm the calendar operation with the user.

## Integration with Other Actions

The Calendar Action is often used in conjunction with other actions to provide a complete interaction flow. For example:

1. User asks to schedule a meeting.
2. Agent uses a Chat Action to gather details about the meeting.
3. Agent uses a Calendar Action to create the meeting event.
4. Agent uses a Chat Action to confirm the meeting details with the user.

This combination of actions allows the agent to provide a seamless and natural interaction experience for the user when managing calendar events.