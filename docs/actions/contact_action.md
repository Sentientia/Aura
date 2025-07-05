# Contact Action

The Contact Action is used to manage contact information. It can retrieve contact details from the user's address book.

## Overview

The Contact Action is implemented in the `ContactAction` class, which extends the `Action` class. It is designed to interact with contact services to retrieve contact information.

## Capabilities

The Contact Action can:

- Retrieve contact information
- Search for contacts by name
- Get recently contacted email addresses
- Format contact details for the agent

## Implementation

The Contact Action is implemented in the `agent/actions/contact_action.py` file. It uses the following components:

- **Action Base Class**: Extends the `Action` class defined in `agent/actions/action.py`.
- **Contact API Integration**: Uses external contact APIs to retrieve contact information.
- **Result Processing**: Processes and formats contact information for the agent.

## Usage

The Contact Action is used by the agent to retrieve contact information. To use the Contact Action:

1. Create a new instance of the `ContactAction` class with the appropriate thought:
   ```python
   from agent.actions.contact_action import ContactAction

   action = ContactAction(thought="I should get the contact information for John Doe")
   ```

2. Execute the action with the current state:
   ```python
   observation = action.execute(state)
   ```

3. The observation will contain the contact information.

## Example

Here's an example of how the Contact Action is used to retrieve contact information:

1. Agent creates a Contact Action:
   ```python
   action = ContactAction(thought="I should get the contact information for John Doe")
   ```

2. Agent executes the action:
   ```python
   observation = action.execute(state)
   ```

3. The action retrieves contact information for John Doe from the user's address book.

4. The observation contains the contact information, which might include:
   - Name
   - Email address
   - Phone number
   - Address
   - Other contact details

5. The agent processes the observation and creates a new action based on the contact information, typically a Chat Action to present the contact information to the user or an Email Action to send an email to the contact.

## Integration with Other Actions

The Contact Action is often used in conjunction with other actions to provide a complete interaction flow. For example:

1. User asks to send an email to John Doe.
2. Agent uses a Contact Action to retrieve John Doe's email address.
3. Agent uses an Email Action to send the email to John Doe.
4. Agent uses a Chat Action to confirm that the email was sent.

This combination of actions allows the agent to provide a seamless and natural interaction experience for the user when working with contacts.