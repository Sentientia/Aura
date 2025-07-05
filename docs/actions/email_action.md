# Email Action

The Email Action is used to send and manage emails. It can compose and send emails on behalf of the user.

## Overview

The Email Action is implemented in the `EmailAction` class, which extends the `Action` class. It is designed to interact with email services to send emails.

## Capabilities

The Email Action can:

- Compose and send emails
- Format email content
- Handle email recipients
- Set email subjects

## Implementation

The Email Action is implemented in the `agent/actions/email_action.py` file. It uses the following components:

- **Action Base Class**: Extends the `Action` class defined in `agent/actions/action.py`.
- **Email API Integration**: Uses external email APIs to send emails.
- **Result Processing**: Processes and formats email operation results for the agent.

## Usage

The Email Action is used by the agent to send emails. To use the Email Action:

1. Create a new instance of the `EmailAction` class with the appropriate thought and payload:
   ```python
   from agent.actions.email_action import EmailAction

   action = EmailAction(
       thought="I should send an email to confirm the meeting",
       payload={
           "to": "recipient@example.com",
           "subject": "Meeting Confirmation",
           "content": "Hello,\n\nThis is to confirm our meeting tomorrow at 2 PM.\n\nBest regards,\nAura"
       }
   )
   ```

2. Execute the action with the current state:
   ```python
   observation = action.execute(state)
   ```

3. The observation will contain the result of the email operation.

## Example

Here's an example of how the Email Action is used to send an email:

1. Agent creates an Email Action:
   ```python
   action = EmailAction(
       thought="I should send an email to the team about the project update",
       payload={
           "to": "team@example.com",
           "subject": "Project Update - July 2025",
           "content": "Hello Team,\n\nHere is the latest update on our project:\n\n- Feature A is complete\n- Feature B is in progress\n- Feature C is scheduled for next week\n\nPlease let me know if you have any questions.\n\nBest regards,\nAura"
       }
   )
   ```

2. Agent executes the action:
   ```python
   observation = action.execute(state)
   ```

3. The action sends an email to team@example.com with the subject "Project Update - July 2025" and the specified content.

4. The observation contains the result of the email operation, which might include:
   - Confirmation that the email was sent
   - Details of the sent email
   - Any errors or warnings that occurred during the operation

5. The agent processes the observation and creates a new action based on the result, typically a Chat Action to confirm the email operation with the user.

## Integration with Other Actions

The Email Action is often used in conjunction with other actions to provide a complete interaction flow. For example:

1. User asks to send an email to the team.
2. Agent uses a Chat Action to gather details about the email.
3. Agent uses an Email Action to send the email.
4. Agent uses a Chat Action to confirm that the email was sent.

This combination of actions allows the agent to provide a seamless and natural interaction experience for the user when sending emails.