"""Offline regression checks: no model, search, or scraping network calls."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ['OPENAI_API_KEY'] = 'offline-test'
os.environ['TAVILY_API_KEY'] = 'offline-test'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import tool
from agents import agents
from pipelines import pipeline
from tools import tools


class ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_result(name, artifact):
    return ToolMessage(content=json.dumps(artifact), artifact=artifact,
                       name=name, tool_call_id='test')


class ResearchTests(unittest.TestCase):
    def test_search_returns_five_complete_candidates(self):
        candidates = [{'title': str(i), 'url': f'https://example.com/{i}',
                       'content': 'evidence ' * 200} for i in range(5)]
        with patch.object(tools.tavily, 'search', return_value={'results': candidates}) as search:
            result = tools.web_search.invoke({'name': 'web_search', 'args': {'query': 'topic'},
                                              'id': 'test', 'type': 'tool_call'})
        search.assert_called_once_with(query='topic', max_results=5)
        self.assertEqual(len(result.artifact), 5)
        self.assertEqual(result.artifact[-1]['snippet'], candidates[-1]['content'])

    def test_agents_stop_repeated_tool_calls(self):
        for builder, tool_name, args in [
            (agents.build_search_agent, 'web_search', {'query': 'topic'}),
            (agents.build_reader_agent, 'scrape_url', {'url': 'https://example.com'})
        ]:
            with self.subTest(tool=tool_name):
                executed = []

                @tool(tool_name)
                def fake_tool(query: str = '', url: str = '') -> str:
                    """Return mock evidence."""
                    executed.append(query or url)
                    return 'Evidence'

                responses = iter(AIMessage(content='', tool_calls=[
                    {'name': tool_name, 'args': args, 'id': str(i)}
                ]) for i in range(10))
                with patch.object(agents, 'llm', ToolCallingModel(messages=responses)), \
                     patch.object(agents, tool_name, fake_tool):
                    result = builder().invoke({'messages': [('human', 'Research topic')]})
                self.assertEqual(len(executed), 2)
                self.assertIn('limit', result['messages'][-1].content.lower())

    def test_pipeline_preserves_evidence_and_passes_it_to_critic(self):
        candidates = [{'title': 'First', 'url': 'https://example.com/a', 'snippet': 'x' * 1200},
                      {'title': 'Last', 'url': 'https://example.com/b', 'snippet': 'important'}]
        extracted = {'url': candidates[-1]['url'], 'status': 'ok', 'content': 'original evidence'}
        search = Mock()
        search.invoke.return_value = {'messages': [
            tool_result('web_search', candidates),
            tool_result('web_search', [candidates[0]]), AIMessage(content='Short summary')]}
        reader = Mock()
        reader.invoke.return_value = {'messages': [tool_result('scrape_url', extracted),
                                                   AIMessage(content='Reader summary')]}
        writer, critic = Mock(), Mock()
        with patch.object(pipeline, 'build_search_agent', return_value=search), \
             patch.object(pipeline, 'build_reader_agent', return_value=reader), \
             patch.object(pipeline, 'writer_chain', writer), \
             patch.object(pipeline, 'critic_chain', critic), contextlib.redirect_stdout(io.StringIO()):
            state = pipeline.run_research_pipeline('topic')
        self.assertEqual(len(state['search_candidates']), 2)
        self.assertIn(candidates[-1]['url'], reader.invoke.call_args.args[0]['messages'][0].content)
        evidence = writer.invoke.call_args.args[0]['research']
        self.assertIn('original evidence', evidence)
        self.assertNotIn('Reader summary', evidence)
        self.assertEqual(critic.invoke.call_args.args[0]['research'], evidence)

    def test_scrape_failure_is_not_article_content(self):
        with patch.object(tools.requests, 'get', side_effect=tools.requests.exceptions.Timeout):
            result = tools.scrape_url.invoke({'name': 'scrape_url',
                'args': {'url': 'https://example.com'}, 'id': 'test', 'type': 'tool_call'})
        self.assertEqual(result.artifact['status'], 'error')
        self.assertEqual(result.artifact['content'], '')
        self.assertEqual(result.artifact['url'], 'https://example.com')


if __name__ == '__main__':
    unittest.main()
