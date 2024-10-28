import os
import asyncio
from python.helpers import memory
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from googlesearch import search  # Import the Google search function

class Knowledge(Tool):
    async def execute(self, question="", **kwargs):
        # Create tasks for Google search and memory search methods
        tasks = [
            self.google_search(question),
            self.mem_search(question)
        ]

        # Run both tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        google_result, memory_result = results

        # Handle exceptions and format results
        google_result = self.format_result(google_result, "Google")
        memory_result = self.format_result(memory_result, "Memory")

        msg = self.agent.read_prompt(
            "tool.knowledge.response.md",
            online_sources=str(google_result),
            memory=memory_result
        )

        await self.agent.handle_intervention(msg)  # Wait for intervention and handle it, if paused

        return Response(message=msg, break_loop=False)

    async def google_search(self, question):
        try:
            # Perform Google search with a specified number of results
            results = search(question, num_results=5)
            return "\n".join(results)
        except Exception as e:
            handle_error(e)
            return f"Google search failed: {str(e)}"

    async def mem_search(self, question: str):
        db = await memory.Memory.get(self.agent)
        docs = await db.search_similarity_threshold(query=question, limit=5, threshold=0.5)
        text = memory.Memory.format_docs_plain(docs)
        return "\n\n".join(text)

    def format_result(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"
        return result if result else ""
