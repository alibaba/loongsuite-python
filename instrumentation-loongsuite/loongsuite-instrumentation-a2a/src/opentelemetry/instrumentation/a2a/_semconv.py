# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental A2A attributes (semantic-conventions-genai PR 195)."""

from urllib.parse import urlsplit

# SDK 0.x and 1.x spell the same protocol operations differently.
CLIENT_METHODS = {
    "send_message": "SendMessage",
    "send_message_streaming": "SendStreamingMessage",
    "get_task": "GetTask",
    "cancel_task": "CancelTask",
    "list_tasks": "ListTasks",
    "resubscribe": "SubscribeToTask",
    "subscribe_to_task": "SubscribeToTask",
    "set_task_callback": "CreateTaskPushNotificationConfig",
    "create_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "get_task_callback": "GetTaskPushNotificationConfig",
    "get_task_push_notification_config": "GetTaskPushNotificationConfig",
    "list_task_push_notification_config": "ListTaskPushNotificationConfigs",
    "list_task_push_notification_configs": "ListTaskPushNotificationConfigs",
    "delete_task_push_notification_config": "DeleteTaskPushNotificationConfig",
    "get_authenticated_extended_card": "GetExtendedAgentCard",
    "get_extended_agent_card": "GetExtendedAgentCard",
}
SERVER_METHODS = {
    "on_message_send": "SendMessage",
    "on_message_send_stream": "SendStreamingMessage",
    "on_get_task": "GetTask",
    "on_cancel_task": "CancelTask",
    "on_list_tasks": "ListTasks",
    "on_resubscribe_to_task": "SubscribeToTask",
    "on_subscribe_to_task": "SubscribeToTask",
    "on_set_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "on_create_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "on_get_task_push_notification_config": "GetTaskPushNotificationConfig",
    "on_list_task_push_notification_config": "ListTaskPushNotificationConfigs",
    "on_list_task_push_notification_configs": "ListTaskPushNotificationConfigs",
    "on_delete_task_push_notification_config": "DeleteTaskPushNotificationConfig",
    "on_get_authenticated_extended_card": "GetExtendedAgentCard",
    "on_get_extended_agent_card": "GetExtendedAgentCard",
}
STREAM_METHODS = {"SendStreamingMessage", "SubscribeToTask"}


def field(obj, name, default=None):
    camel = name.split("_")[0] + "".join(
        p.title() for p in name.split("_")[1:]
    )
    if isinstance(obj, dict):
        return obj.get(name, obj.get(camel, default))
    # Protobuf message fields return empty objects even when unset.
    has_field = getattr(obj, "HasField", None)
    if has_field is not None:
        try:
            if not has_field(name):
                return default
        except ValueError:
            pass  # Repeated and scalar fields do not support presence.
    return getattr(obj, name, getattr(obj, camel, default))


def payload(value):
    for name in ("root", "result", "params"):
        nested = field(value, name)
        if nested is not None:
            value = nested
    return value


def state_name(status):
    state = field(status, "state")
    if isinstance(state, int):
        descriptor = getattr(status, "DESCRIPTOR", None)
        state_field = (
            descriptor.fields_by_name.get("state") if descriptor else None
        )
        enum = state_field.enum_type if state_field else None
        value = enum.values_by_number.get(state) if enum else None
        return value.name if value else None
    state = getattr(state, "value", state)
    if state:
        text = str(state).upper().replace("-", "_")
        return text if text.startswith("TASK_STATE_") else "TASK_STATE_" + text
    return None


def attributes(value, *, request=False):
    value = payload(value)
    message = field(value, "message")
    task = field(value, "task")
    event = field(value, "status_update") or field(value, "artifact_update")
    item = message or task or event or value
    attrs = {}
    context_id = field(item, "context_id")
    task_id = field(item, "task_id")
    status = field(item, "status")
    if status is not None:
        task_id = task_id or field(item, "id")
        state = state_name(status)
        if state:
            attrs["a2a.task.state"] = state
    if request and message is None and not field(value, "jsonrpc"):
        task_id = task_id or field(item, "id")
    if task_id:
        attrs["a2a.task.id"] = task_id
    if context_id:
        attrs["gen_ai.conversation.id"] = context_id
    if request:
        message_id = field(item, "message_id")
        if message_id:
            attrs["a2a.message.id"] = message_id
        references = field(item, "reference_task_ids")
        if references:
            attrs["a2a.message.reference_task_ids"] = list(references)
        tenant = field(value, "tenant")
        if tenant:
            attrs["a2a.tenant"] = tenant
    return attrs


def card_attributes(instance):
    card = getattr(instance, "agent_card", None) or getattr(
        instance, "_agent_card", None
    )
    attrs = {}
    for key in ("name", "description", "version"):
        value = field(card, key)
        if value:
            attrs["gen_ai.agent." + key] = value
    url = getattr(instance, "url", None) or getattr(instance, "_url", None)
    if url:
        parsed = urlsplit(url)
        if parsed.hostname:
            attrs["server.address"] = parsed.hostname
            attrs["server.port"] = parsed.port or (
                443 if parsed.scheme == "https" else 80
            )
        # Only the selected endpoint's interface can identify a protocol version.
        for interface in field(card, "supported_interfaces", ()):
            if field(interface, "url") == url:
                version = field(interface, "protocol_version")
                if version:
                    attrs["a2a.protocol.version"] = version
                break
    version = field(card, "protocol_version")
    if version:
        attrs["a2a.protocol.version"] = version
    return attrs


def terminal(value):
    value = payload(value)
    if field(value, "message") is not None or field(value, "message_id"):
        return True
    event = field(value, "status_update") or field(value, "task") or value
    if field(event, "final", False):
        return True
    status = field(event, "status")
    return status is not None and state_name(status) in {
        "TASK_STATE_COMPLETED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_FAILED",
        "TASK_STATE_REJECTED",
    }
