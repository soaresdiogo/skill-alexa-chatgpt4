import copy
import logging
import os
import re

import ask_sdk_core.utils as ask_utils

from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_model import Response

from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Padrão para o repositório; na Lambda configure OPENAI_API_KEY (não commite a chave real).
_OPENAI_KEY_PLACEHOLDER = "SUBSTITUA-POR-SUA-API-KEY-DA-OPENAI"
openai_api_key = os.environ.get("OPENAI_API_KEY", _OPENAI_KEY_PLACEHOLDER)

client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4o-mini"
SESSION_MESSAGES_KEY = "chat_messages"
MAX_DIALOG_TURNS = 8
MAX_SPEECH_CHARS = 4000

_SYSTEM_MESSAGE = {
    "role": "system",
    "content": "Você é uma assistente muito útil. Por favor, responda de forma clara e concisa em Português do Brasil.",
}


def _trim_history(messages, max_user_assistant_pairs=MAX_DIALOG_TURNS):
    # type: (list, int) -> list
    cap = 1 + (max_user_assistant_pairs * 2)
    if len(messages) <= cap:
        return messages
    return [messages[0]] + messages[-(max_user_assistant_pairs * 2) :]


def _get_or_init_messages(handler_input):
    # type: (HandlerInput) -> list
    attrs = handler_input.attributes_manager.session_attributes
    existing = attrs.get(SESSION_MESSAGES_KEY)
    if not existing or not isinstance(existing, list):
        return [copy.deepcopy(_SYSTEM_MESSAGE)]
    return copy.deepcopy(existing)


def _set_messages(handler_input, messages):
    # type: (HandlerInput, list) -> None
    handler_input.attributes_manager.session_attributes[SESSION_MESSAGES_KEY] = copy.deepcopy(
        messages
    )


def _for_alexa_speech(text, max_len=MAX_SPEECH_CHARS):
    # type: (object, int) -> str
    if text is None or (isinstance(text, str) and not text.strip()):
        return "Não obtive resposta. Pode tentar de novo?"
    s = re.sub(r"\s+", " ", str(text).strip())
    s = s.replace("&", " e ")
    s = s.replace("<", " ")
    s = s.replace(">", " ")
    if len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


def _slot_query_value(handler_input):
    # type: (HandlerInput) -> str
    try:
        intent = handler_input.request_envelope.request.intent
    except (AttributeError, TypeError):
        return ""
    if intent is None or not intent.slots:
        return ""
    slot = intent.slots.get("query")
    if slot is None or slot.value is None:
        return ""
    return str(slot.value).strip()


def generate_gpt_response(handler_input, user_text):
    # type: (HandlerInput, str) -> str
    history = _get_or_init_messages(handler_input)
    messages = history + [{"role": "user", "content": user_text}]
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=700, temperature=0.8
        )
        reply = response.choices[0].message.content
        assistant_text = (
            reply
            if reply is not None
            else "Não obtive resposta. Pode tentar de novo?"
        )
        messages = messages + [{"role": "assistant", "content": assistant_text}]
        _set_messages(handler_input, _trim_history(messages))
        return _for_alexa_speech(assistant_text)
    except Exception:
        logger.exception("Falha na chamada OpenAI")
        return _for_alexa_speech("Erro ao gerar resposta. Tente de novo em instantes.")


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _set_messages(handler_input, [copy.deepcopy(_SYSTEM_MESSAGE)])
        speak_output = (
            "Bem vindo ao Chat 'Gepetê Quatro' da 'Open ei ai'! Qual a sua pergunta?"
        )
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class GptQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        query = _slot_query_value(handler_input)
        if not query:
            reprompt = "Não entendi a pergunta. Pode repetir?"
            return (
                handler_input.response_builder.speak(reprompt)
                .ask(reprompt)
                .response
            )
        response = generate_gpt_response(handler_input, query)
        return (
            handler_input.response_builder.speak(response)
            .ask("Você pode fazer uma nova pergunta ou falar: sair.")
            .response
        )


class AmazonFallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        reprompt = "Não entendi. Pode fazer a pergunta de outra forma?"
        return (
            handler_input.response_builder.speak(reprompt)
            .ask(reprompt)
            .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Como posso te ajudar?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.CancelIntent")(
            handler_input
        ) or ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Até logo!"
        return handler_input.response_builder.speak(speak_output).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)
        speak_output = "Desculpe, não consegui processar sua solicitação."
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(AmazonFallbackIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
