import os
import asyncio
from python.helpers import dotenv
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.searxng import search as searxng

SEARCH_ENGINE_RESULTS = 10
black_list_keyword = ["youtube", "facebook"]  # ব্ল্যাকলিস্টেড কিওয়ার্ড লিস্ট

class Knowledge(Tool):
    async def execute(self, question="", **kwargs):
        # SearxNG থেকে URL সংগ্রহ করা
        searxng_result = await self.searxng_search(question)
        urls = self.format_result_searxng(searxng_result, "Search Engine")

        msg = self.agent.read_prompt(
            "tool.knowledge.response.md",
            online_sources=urls if urls else "",
        )

        await self.agent.handle_intervention(msg)  # ইন্টারভেনশন হ্যান্ডেল করা

        return Response(message=msg, break_loop=False)

    async def searxng_search(self, question):
        return await searxng(question)

    def format_result_searxng(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"

        urls = []
        for item in result["results"]:
            if "url" in item:
                url = item["url"]
                # ব্ল্যাকলিস্টেড কিওয়ার্ড চেক করা
                if not any(keyword in url for keyword in black_list_keyword):
                    urls.append(url)

        return "\n".join(urls[:SEARCH_ENGINE_RESULTS]).strip()
