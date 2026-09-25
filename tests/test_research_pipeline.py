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

from langchain_core.messages import AIMessage, ToolMessage
from agents import agents
from pipelines import pipeline
from tools import tools


def tool_result(name, artifact):
    return ToolMessage(content=json.dumps(artifact), artifact=artifact,
                       name=name, tool_call_id='test')


class ResearchTests(unittest.TestCase):
    def test_search_preserves_urls_and_snippets(self):
        candidates = [{'title': str(i), 'url': f'https://example.com/{i}',
                       'content': 'evidence ' * 200} for i in range(1)]
        with patch.object(tools.tavily, 'search', return_value={'results': candidates}) as search:
            result = tools.web_search.invoke({'name': 'web_search', 'args': {'query': 'topic'},
                                              'id': 'test', 'type': 'tool_call'})
        search.assert_called_once_with(query='topic', max_results=1)
        self.assertEqual(len(result.artifact), 1)
        self.assertEqual(result.artifact[-1]['snippet'], candidates[-1]['content'])
        self.assertEqual(result.artifact[-1]['url'], candidates[-1]['url'])

    def test_pipeline_preserves_evidence_and_passes_it_to_critic(self):
        candidates = [{'title': 'First', 'url': 'https://example.com/a', 'snippet': 'x' * 1200},
                      {'title': 'Last', 'url': 'https://example.com/b', 'snippet': 'important'}]
        extracted = {'url': candidates[-1]['url'], 'status': 'ok', 'content': 'original evidence'}
        search = Mock()
        search.invoke.return_value = {'messages': [
            tool_result('web_search', candidates),
            tool_result('web_search', [candidates[0]]), AIMessage(content='Short summary')]}
        reader = Mock()
        unselected = {'url': candidates[0]['url'], 'status': 'ok', 'content': 'unselected article'}
        reader.invoke.return_value = {
            'messages': [tool_result('scrape_url', extracted),
                         tool_result('scrape_url', unselected), AIMessage(content='Reader summary')],
            'structured_response': agents.ReaderSelection(
                selected_url=extracted['url'], reason='Most relevant evidence')}
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
        self.assertNotIn('unselected article', evidence)
        self.assertNotIn(candidates[0]['url'], evidence)
        self.assertEqual(state['selected_source'], extracted)
        self.assertEqual(len(state['scrape_results']), 2)
        self.assertEqual(critic.invoke.call_args.args[0]['research'], evidence)

    def test_selection_rejects_missing_invalid_or_failed_sources(self):
        candidates = [{'url': 'https://example.com/a'}]
        for url, status, content in [
            (None, 'ok', 'text'),
            ('https://example.com/unknown', 'ok', 'text'),
            (candidates[0]['url'], 'error', 'text'),
            (candidates[0]['url'], 'ok', '   '),
        ]:
            with self.subTest(url=url, status=status, content=content):
                result = {
                    'messages': [tool_result('scrape_url', {
                        'url': url, 'status': status, 'content': content})],
                    'structured_response': agents.ReaderSelection(selected_url=url, reason='test')}
                with self.assertRaises(RuntimeError):
                    pipeline.selected_reader_evidence(result, candidates)
        with self.assertRaises(RuntimeError):
            pipeline.selected_reader_evidence({'messages': []}, candidates)

    def test_article_excerpt_is_limited_and_marked_truncated(self):
        response = Mock(text='<html>test</html>')
        with patch.object(tools.requests, 'get', return_value=response), \
             patch.object(tools.trafilatura, 'extract', return_value='x' * 6000):
            result = tools._extract_url('https://example.com/article')
        self.assertEqual(len(result['content']), 5000)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['url'], 'https://example.com/article')

    def test_scrape_failure_is_not_article_content(self):
        with patch.object(tools.requests, 'get', side_effect=tools.requests.exceptions.Timeout):
            result = tools.scrape_url.invoke({'name': 'scrape_url',
                'args': {'url': 'https://example.com'}, 'id': 'test', 'type': 'tool_call'})
        self.assertEqual(result.artifact['status'], 'error')
        self.assertEqual(result.artifact['content'], '')
        self.assertEqual(result.artifact['url'], 'https://example.com')


if __name__ == '__main__':
    unittest.main()
