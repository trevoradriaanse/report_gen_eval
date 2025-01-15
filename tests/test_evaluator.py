from report_gen_eval import ModelProvider
from report_gen_eval.evaluator import empty_response, check_citations_relevance_detail, process_w_citations, \
    process_citation_relevancy, load_nugget
from report_gen_eval.utils import load_jsonl


def test_empty_response():
    assert (empty_response('sent') ==
            {"segment_type": 'sentence',
             "text": 'sent',
             "citations": [],
             "judgments": [],
             })


def test_check_citations_relevance_detail_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'}],
                                            ModelProvider.YES) == ['RELEVANT']


def test_check_citations_relevance_detail_not_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'}],
                                            ModelProvider.NO) == ['NOT_RELEVANT']


def test_check_citations_relevance_detail_many_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'},
                                            {"doc_id": "D2", "text": 'this is another citation'},
                                            {"doc_id": "D3", "text": 'this is yet another test citation'}],
                                            ModelProvider.YES) == ['RELEVANT', 'RELEVANT', 'RELEVANT']


def test_process_citation_relevancy():
    assert process_citation_relevancy([{"doc_id": "D1", "text":'one citation'},
                                       {"doc_id": "D2", "text":'another citation'},
                                       {"doc_id": "D3", "text":'yet another citation'}],
                                      '',
                                      ModelProvider.YES,
                                      empty_response('this is a test sentence'),
                                      'this is a test sentence'
                                      ) == {
                                      "segment_type": "sentence",
                                      "text": 'this is a test sentence',
                                      "citations": [],
                                      "judgments": [
                                          {
                                              "judgment_type_id": "Cited document is relevant?",
                                              "response": {"docid": "D1", "answer": "RELEVANT",},
                                              "evaluator": ModelProvider.YES,
                                              "provenance": None,
                                          },
                                          {
                                              "judgment_type_id": "Cited document is relevant?",
                                              "response": {"docid": "D2", "answer": "RELEVANT",},
                                              "evaluator": ModelProvider.YES,
                                              "provenance": None,
                                          },
                                          {
                                              "judgment_type_id": "Cited document is relevant?",
                                              "response": {"docid": "D3", "answer": "RELEVANT",},
                                              "evaluator": ModelProvider.YES,
                                              "provenance": None,
                                          },
                                          ],
    }


def test_load_nugget():
    assert load_nugget('assets/avengers_nuggets_fix.jsonl',
                       load_jsonl('assets/avengers_report.jsonl'),
                       False) is not None
