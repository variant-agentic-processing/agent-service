"""Claude reasoning loop — runs tool calls against the MCP server."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from src.mcp_client import mcp_session, truncate_result

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a genomic variant analysis assistant for a research prototype.
You have access to a ClickHouse database containing:
  - variants table: per-individual genotype data, GRCh38, chr-prefixed chromosomes
  - annotations table: ClinVar annotations (release 2025-03), joined at query time on (chromosome, position, ref, alt)

Do not assume which individuals are loaded. Use list_individuals to find available individual IDs before querying.

Available tools:
  - list_individuals: List all individual IDs currently in the database with variant counts. Call this to confirm an ID exists before querying variants.
  - list_samples: List all ingested individuals with full metadata (display name, sex, population, superpopulation). Use this when the user refers to individuals by name, sex, or ancestry rather than by ID.
  - get_sample: Fetch display name, sex, and population metadata for a single individual. Use this to answer "who is HG00096?" or to enrich variant results with identity context.
  - describe_schema: Get field names, valid filter values, and example calls. Call this first if uncertain about any field name or valid value. Do not guess.
  - get_individual_summary: Overall variant burden for one individual. Use for "how many variants", "what genes are affected" questions.
  - search_variants: Find variants for one individual matching filters (gene, chromosome, position range, clinical_significance). Requires individual_id plus at least one filter. Use limit=200 for targeted queries (specific gene or significance); limit=20 is only appropriate for exploratory broad searches.
  - query_by_locus: Find all individuals carrying variants in a genomic region. For cross-individual questions. Max window 1,000,000 bases.
  - aggregate_cohort: Population-level counts grouped by a field (gene_symbol, clinical_significance, chromosome, consequence, review_status).
  - annotation_lookup: Full ClinVar record for a specific variant by rsID or exact coordinates.

Tool routing guidance:
  - "What individuals/samples are available?" → list_individuals
  - "Who is HG00096?" / enrich result with name or demographics → get_sample
  - User refers to individual by name, sex, or ancestry → list_samples to find their ID first
  - Individual burden/summary → get_individual_summary
  - Individual variants with filters → search_variants
  - Cross-individual locus/region → query_by_locus
  - Population statistics → aggregate_cohort
  - Specific variant annotation → annotation_lookup
  - Uncertain about fields/values → describe_schema

Response format:
  - Plain text only. No markdown. No headers, no bullet points with dashes, no bold, no tables, no code blocks.
  - Use short paragraphs and blank lines to separate distinct ideas or data points.
  - When listing multiple items (variants, genes, individuals), put each on its own line with a simple label, e.g. "BRCA1: 3 pathogenic variants".
  - Keep responses conversational and concise. Lead with the direct answer, then supporting detail.
  - Whenever an individual_id appears in a response, always pair it with the display name in the format "Display Name (HG00096)". If you do not already have the display name, call get_sample to retrieve it before responding. Never mention a bare individual_id without the display name.

Clinical accuracy requirements:
  - Always attribute clinical significance to ClinVar. Never state your own clinical judgement.
  - Add "according to ClinVar" to all pathogenicity statements.
  - Append: "This is a research prototype using public 1000 Genomes data. Not for clinical use."

Handling limitations:
  - If a result is marked [TRUNCATED] and the query was targeted (specific gene, rsID, or significance), retry with a higher limit before responding. Only offer to narrow the query if the result is still truncated after increasing the limit.
  - Cross-individual queries (query_by_locus, aggregate_cohort) may be slower — warn the user if results take time.
  - If the question cannot be answered from the available data, say so clearly. Do not hallucinate variants or annotations.
"""


async def run(
    messages: list[dict],
    tools_schemas: list[dict],
    anthropic_client: anthropic.AsyncAnthropic,
    model: str,
    context: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the agent reasoning loop. Yields SSE-compatible event dicts."""
    system = SYSTEM_PROMPT + f"\n\n{context}" if context else SYSTEM_PROMPT
    messages = list(messages)  # don't mutate caller's list

    async with mcp_session() as session:
        for iteration in range(MAX_ITERATIONS):
            logger.info("Agent iteration %d, messages=%d", iteration + 1, len(messages))

            response = await anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tools_schemas,
            )

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                # No tool calls — extract final text answer
                text_blocks = [b for b in response.content if b.type == "text"]
                final_text = "\n".join(b.text for b in text_blocks).strip()
                yield {"type": "answer", "text": final_text}
                return

            # Execute all tool calls in this turn
            tool_results = []
            for tool_use in tool_uses:
                yield {
                    "type": "tool_call",
                    "tool": tool_use.name,
                    "args": tool_use.input,
                }

                try:
                    result = await session.call_tool(tool_use.name, tool_use.input)
                    result_text = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    result_text = truncate_result(result_text)
                    is_error = result.isError or False
                except Exception as exc:
                    result_text = f"Tool call failed: {exc}"
                    is_error = True

                yield {
                    "type": "tool_result",
                    "tool": tool_use.name,
                    "chars": len(result_text),
                    "is_error": is_error,
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_text,
                    **({"is_error": True} if is_error else {}),
                })

            messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "error",
        "message": f"Max iterations ({MAX_ITERATIONS}) reached without a final answer.",
    }
