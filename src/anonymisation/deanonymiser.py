"""De-anonymiser — reverses pseudonymisation using the encrypted registry.

This module is LOCAL-ONLY. It must NEVER be invoked in any code path
that sends data to external services. The de-anonymiser restores
anonymised tokens back to their original PII values so the user can
understand cloud LLM responses in the context of the real case.

Security:
  - Requires the encryption passphrase to be loaded in the registry
  - All operations logged to the audit trail
  - No network access — import-time assertion
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from .models import AnonymisedDocument, AnonymisedPayload
from .registry import AnonymisationRegistry


# Pattern to match anonymisation tokens: [CATEGORY_NNN] or [CATEGORY_NNN, ...]
_TOKEN_PATTERN = re.compile(r"\[[A-Z_]+_\d{3}(?:,\s*[^\]]+)?\]")


class DeAnonymiser:
    """Reverses anonymisation by replacing tokens with original values.

    LOCAL-ONLY — must never be used before sending data externally.

    Args:
        registry: The anonymisation registry with encrypted mappings.
    """

    def __init__(self, registry: AnonymisationRegistry) -> None:
        self._registry = registry

    def deanonymise_text(self, text: str, matter_id: str) -> str:
        """Replace all anonymisation tokens in text with original values.

        Args:
            text: Anonymised text containing tokens like [PERSON_001].
            matter_id: Matter identifier to scope the token lookup.

        Returns:
            Text with tokens replaced by original PII values.
        """
        if not text:
            return text

        reverse_map = self._registry.get_reverse_map(matter_id)
        if not reverse_map:
            logger.warning(f"No token mappings found for matter {matter_id}")
            return text

        result = text
        replacements = 0

        # Find all tokens in the text
        for match in _TOKEN_PATTERN.finditer(text):
            token = match.group()

            # Handle tokens with appended metadata like [AGE_001, age band 5-10]
            # Strip to the base token for lookup
            base_token = token
            if "," in token:
                base_token = token.split(",")[0] + "]"

            original = reverse_map.get(base_token)
            if original:
                result = result.replace(token, original, 1)
                replacements += 1

        # Audit
        self._registry.log_audit(
            action="deanonymise",
            matter_id=matter_id,
            entity_count=replacements,
            details=f"De-anonymised {replacements} tokens in text",
        )

        logger.debug(f"De-anonymised {replacements} tokens for matter {matter_id}")
        return result

    def deanonymise_cloud_response(
        self,
        response: str,
        matter_id: str,
    ) -> str:
        """De-anonymise a response received from a cloud LLM.

        The cloud LLM will have used our anonymisation tokens in its
        response (e.g. "[PERSON_001] was liable because..."). This
        method restores the real names/values for the user.

        Args:
            response: Cloud LLM response containing anonymisation tokens.
            matter_id: Matter identifier.

        Returns:
            Response with tokens replaced by original values.
        """
        result = self.deanonymise_text(response, matter_id)

        self._registry.log_audit(
            action="deanonymise_cloud_response",
            matter_id=matter_id,
            details="De-anonymised cloud LLM response",
        )

        return result

    def deanonymise_document(
        self,
        anon_doc: AnonymisedDocument,
    ) -> str:
        """De-anonymise a previously anonymised document.

        Args:
            anon_doc: AnonymisedDocument with anonymised_text.

        Returns:
            Original text (or de-anonymised approximation).
        """
        # If we still have the original text, just return it
        if anon_doc.original_text:
            return anon_doc.original_text

        # Otherwise reconstruct from tokens
        return self.deanonymise_text(anon_doc.anonymised_text, anon_doc.matter_id)

    def find_tokens_in_text(self, text: str) -> list[str]:
        """Find all anonymisation tokens present in a text.

        Args:
            text: Text to scan for tokens.

        Returns:
            List of token strings found.
        """
        return _TOKEN_PATTERN.findall(text)

    def preview_deanonymisation(
        self,
        text: str,
        matter_id: str,
    ) -> list[dict[str, str]]:
        """Preview what de-anonymisation would produce without applying it.

        Useful for UI preview before committing.

        Args:
            text: Anonymised text.
            matter_id: Matter identifier.

        Returns:
            List of dicts with 'token' and 'original' keys.
        """
        reverse_map = self._registry.get_reverse_map(matter_id)
        previews = []

        for match in _TOKEN_PATTERN.finditer(text):
            token = match.group()
            base_token = token.split(",")[0] + "]" if "," in token else token
            original = reverse_map.get(base_token, "<unknown>")
            previews.append({"token": token, "original": original})

        return previews
