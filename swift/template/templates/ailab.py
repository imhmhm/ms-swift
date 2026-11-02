# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import deepcopy

from ..base import Template
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs


class AilabSlmRerankerTemplate(Template):
    """Generative reranker template on the ailab_slm chat format.

    Reranker rows keep the candidate document in the assistant slot as a
    container convention. This template folds the document back into the query
    by substituting the `{doc_message}` placeholder (same design as
    Qwen3RerankerTemplate merging query and document into a single user turn),
    so the encoded sequence ends at the assistant-slot header where the yes/no
    next-token logits are scored.
    """

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        raw_messages = deepcopy(inputs.messages)
        super()._preprocess_inputs(inputs)
        query = raw_messages[0]['content'] if len(raw_messages) > 0 else ''
        document = raw_messages[1]['content'] if len(raw_messages) > 1 else ''
        assert '{doc_message}' in query, 'query must contain the "{doc_message}" placeholder'
        inputs.messages = [{'role': 'user', 'content': query.replace('{doc_message}', document)}]
        return inputs


register_template(
    TemplateMeta(
        'ailab_slm',
        prefix=[],
        prompt=['[extra_id_254]user\n{{QUERY}}[extra_id_255]\n[extra_id_254]assistant\n'],
        chat_sep=['[extra_id_255]\n'],
        suffix=['[extra_id_255]\n'],
        system_prefix=['[extra_id_254]system\n{{SYSTEM}}[extra_id_255]\n'],
        default_system='You are a helpful assistant.',
        stop_words=['[extra_id_255]']))

register_template(
    TemplateMeta(
        'ailab_slm_reranker',
        prefix=[],
        prompt=['[extra_id_254]user\n{{QUERY}}[extra_id_255]\n[extra_id_254]assistant\n'],
        chat_sep=['[extra_id_255]\n'],
        suffix=['[extra_id_255]\n'],
        system_prefix=['[extra_id_254]system\n{{SYSTEM}}[extra_id_255]\n'],
        default_system='You are a helpful assistant.',
        stop_words=['[extra_id_255]'],
        template_cls=AilabSlmRerankerTemplate))
